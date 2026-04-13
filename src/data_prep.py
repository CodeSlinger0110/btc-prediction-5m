"""
Load Binance 1s klines, discover valid 5-minute windows, and materialize (X, y, meta).
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from config import Config, FeatureConfig
from features import (
    compute_pre_window_stats,
    precompute_indicators,
    precompute_indicators_chunked,
    vectorized_window_feature_block,
)


# Binance aggTrades/klines-style CSV without header (1s kline).
BINANCE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

# Training only needs the first six columns (open_time … volume). Skipping the rest
# cuts CSV RAM roughly in half versus reading all 12 Binance fields.
BINANCE_LOAD_COLS = BINANCE_COLS[:6]
_CSV_DTYPES_LOAD: Dict[str, type] = {
    "open_time": np.int64,
    "open": np.float32,
    "high": np.float32,
    "low": np.float32,
    "close": np.float32,
    "volume": np.float32,
}


def _parquet_engine_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _read_binance_csv_ohlcv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=BINANCE_LOAD_COLS,
        usecols=list(range(6)),
        dtype=_CSV_DTYPES_LOAD,
        engine="c",
    )


def _merge_raw_csv_files_on_disk(paths: List[Path], dest: Path) -> None:
    """
    Append raw CSV bytes in ``paths`` order (no pandas). Peak RAM is O(1); needs free disk
    space ~sum(file sizes) for ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = 8 * 1024 * 1024
    with open(dest, "wb") as out_f:
        for p in paths:
            with open(p, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=buf)


def _materialize_klines_from_parquet_shards(
    part_paths: List[Path],
    *,
    log: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    """
    Copy shard Parquet files into one contiguous structured array, sort, dedupe.

    Avoids merging shards into one giant Parquet file then ``pd.read_parquet``, which loads
    the entire dataset into RAM twice (Arrow decode + pandas) and often OOM-kills the process.
    """
    import pyarrow.parquet as pq

    if not part_paths:
        raise ValueError("part_paths must be non-empty")

    metas: List[int] = []
    for p in part_paths:
        metas.append(int(pq.ParquetFile(p).metadata.num_rows))
    total = sum(metas)
    if total == 0:
        raise RuntimeError("Parquet shards contain zero rows.")

    if log:
        log(
            f"  materializing {len(part_paths)} shard(s) → {total:,} rows "
            "(preallocated; no single huge read_parquet) …"
        )

    dt = np.dtype(
        [
            ("open_time", np.int64),
            ("open", np.float64),
            ("high", np.float64),
            ("low", np.float64),
            ("close", np.float64),
            ("volume", np.float64),
        ]
    )
    block = np.empty(total, dtype=dt)
    off = 0
    n_sh = len(part_paths)
    cols = BINANCE_LOAD_COLS
    for i, p in enumerate(part_paths, start=1):
        if log and (i == 1 or i == n_sh or (n_sh > 10 and i % max(1, n_sh // 10) == 0)):
            log(f"  copy shard {i}/{n_sh} ({p.name}) → offset {off:,} …")
        tbl = pq.read_table(p, columns=cols, memory_map=True)
        n = int(tbl.num_rows)
        block["open_time"][off : off + n] = np.asarray(tbl.column("open_time"), dtype=np.int64)
        for cname in ("open", "high", "low", "close", "volume"):
            block[cname][off : off + n] = np.asarray(tbl.column(cname), dtype=np.float64)
        off += n
        del tbl
        try:
            p.unlink()
        except OSError:
            pass
        gc.collect()

    if off != total:
        raise RuntimeError(f"Row count mismatch after materialize: {off} vs {total}")

    if log:
        log(f"  sorting {total:,} rows by open_time (in-place) …")
    t_s = time.perf_counter()
    block.sort(order="open_time")
    if log:
        log(f"  sort done in {time.perf_counter() - t_s:.1f}s.")

    if log:
        log("  deduplicating open_time (keep last) …")
    t = block["open_time"]
    if len(t):
        mask = np.r_[t[:-1] != t[1:], True]
        block = block[mask]

    if log:
        log(f"  ready: {len(block):,} rows after dedupe.")

    names = list(dt.names)
    return pd.DataFrame({n: block[n] for n in names})


def _sort_dedupe_cast_klines(
    out: pd.DataFrame,
    *,
    t_load: float,
    t_merge: float,
    log: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    if log:
        log(
            f"  table ready in {time.perf_counter() - t_merge:.1f}s → {len(out):,} rows; "
            "final sort/dedupe check …"
        )
    t_sort = time.perf_counter()
    out["open_time"] = out["open_time"].astype(np.int64)
    # Shards appended in sorted filename order are usually already time-ordered; skipping
    # sort_values avoids a large extra allocation when monotonic.
    if out["open_time"].is_monotonic_increasing:
        if log:
            log("  open_time already monotonic; skipping sort_values …")
        out = out.drop_duplicates(subset=["open_time"], keep="last")
    else:
        out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    out = out.reset_index(drop=True)
    if log:
        log(f"  sort/dedupe {time.perf_counter() - t_sort:.1f}s → {len(out):,} unique rows")

    for c in ("open", "high", "low", "close", "volume"):
        out[c] = out[c].astype(np.float64)

    gc.collect()
    if log:
        log(f"  raw CSV stage done in {time.perf_counter() - t_load:.1f}s total.")
    return out


_PARQUET_COMPRESSION = "zstd"


def load_binance_1s_csvs(
    data_dir: Path,
    glob_pattern: str,
    *,
    log: Optional[Callable[[str], None]] = None,
    read_wave_size: int = 1,
    merge_batch_size: int = 2,
    use_disk_spill: bool = True,
    parquet_merge_strategy: str = "sequential",
) -> pd.DataFrame:
    """Load and concatenate all matching CSVs, sorted by open_time."""
    wave = max(1, int(read_wave_size))
    t_load = time.perf_counter()
    paths = sorted(data_dir.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {glob_pattern} under {data_dir}")

    total_bytes = sum(p.stat().st_size for p in paths)
    pa_ok = _parquet_engine_available()
    # Multiple CSVs: always use disk spill when PyArrow is available (never hold all frames in RAM).
    spill = pa_ok and (bool(use_disk_spill) or len(paths) > 1)

    if log:
        log(
            f"  {len(paths)} file(s), ~{total_bytes / (1024 ** 3):.2f} GiB on disk — reading …"
        )
        if spill:
            log(
                f"  load plan: wave={wave} → one Parquet shard per CSV, "
                "then preallocated NumPy materialize (no merge-to-one-giant-Parquet read)."
            )
        else:
            if use_disk_spill and not pa_ok:
                log(
                    "  pyarrow not installed: will merge raw CSV bytes on disk, then one read_csv "
                    "(install pyarrow for parquet spill — usually faster and lower peak RAM)."
                )
            log(
                "  merge: stream-append CSVs to temp file on disk → single read "
                "(no pandas concat of many DataFrames)."
            )

    t_merge = time.perf_counter()

    # One CSV: read directly — avoids writing/reading temp parquet (saves RAM + I/O).
    if len(paths) == 1:
        p = paths[0]
        t_file = time.perf_counter()
        if log:
            log(f"  [1/1] {p.name} (single file — no temp parquet merge) …")
        out = _read_binance_csv_ohlcv(p)
        if log:
            log(f"      {len(out):,} rows in {time.perf_counter() - t_file:.1f}s")
        return _sort_dedupe_cast_klines(out, t_load=t_load, t_merge=t_merge, log=log)

    if spill:
        with tempfile.TemporaryDirectory(prefix="btc_klines_") as tmp:
            tmp_path = Path(tmp)
            part_paths: List[Path] = []
            buf: List[pd.DataFrame] = []

            # Never concat several full CSV DataFrames in RAM when many files are present.
            if len(paths) > 1 and wave > 1:
                prev_wave = wave
                if log:
                    log(
                        f"  forcing csv_read_wave_size to 1 (config had {prev_wave}); "
                        "wave>1 would pd.concat multiple CSVs in RAM and often OOMs."
                    )
                wave = 1

            # wave==1: one CSV → one parquet shard (no in-RAM concat of multiple CSVs).
            if wave == 1:
                for i, p in enumerate(paths, start=1):
                    t_file = time.perf_counter()
                    if log:
                        log(f"  [{i}/{len(paths)}] {p.name} …")
                    df = _read_binance_csv_ohlcv(p)
                    if log:
                        log(f"      {len(df):,} rows in {time.perf_counter() - t_file:.1f}s")
                    part_path = tmp_path / f"part_{len(part_paths)}.parquet"
                    df.to_parquet(part_path, index=False, compression=_PARQUET_COMPRESSION)
                    del df
                    part_paths.append(part_path)
                    gc.collect()
            else:
                for i, p in enumerate(paths, start=1):
                    t_file = time.perf_counter()
                    if log:
                        log(f"  [{i}/{len(paths)}] {p.name} …")
                    df = _read_binance_csv_ohlcv(p)
                    if log:
                        log(f"      {len(df):,} rows in {time.perf_counter() - t_file:.1f}s")
                    buf.append(df)
                    if len(buf) >= wave:
                        block = pd.concat(buf, ignore_index=True, copy=False)
                        buf.clear()
                        part_path = tmp_path / f"part_{len(part_paths)}.parquet"
                        block.to_parquet(part_path, index=False, compression=_PARQUET_COMPRESSION)
                        del block
                        part_paths.append(part_path)
                        gc.collect()

                if buf:
                    block = pd.concat(buf, ignore_index=True, copy=False)
                    buf.clear()
                    part_path = tmp_path / f"part_{len(part_paths)}.parquet"
                    block.to_parquet(part_path, index=False, compression=_PARQUET_COMPRESSION)
                    del block
                    part_paths.append(part_path)
                    gc.collect()

            if log:
                log(
                    f"  {len(part_paths)} parquet shard(s); materializing into one table "
                    "(preallocated RAM — avoids merging to one huge Parquet + read_parquet OOM) …"
                )
            out = _materialize_klines_from_parquet_shards(part_paths, log=log)
    else:
        # Without PyArrow we cannot write parquet shards. Never build a list of full DataFrames
        # and pd.concat them — that OOMs on many/large files. Stream-merge CSV bytes on disk.
        with tempfile.TemporaryDirectory(prefix="btc_klines_csvmerge_") as tmp:
            merged = Path(tmp) / "merged.csv"
            if log:
                log(
                    f"  streaming {len(paths)} CSV(s) → {merged.name} (~{total_bytes / (1024 ** 3):.2f} GiB; "
                    "ensure enough free disk) …"
                )
            t_disk = time.perf_counter()
            _merge_raw_csv_files_on_disk(paths, merged)
            if log:
                log(f"  disk append done in {time.perf_counter() - t_disk:.1f}s; read_csv …")
            out = _read_binance_csv_ohlcv(merged)
            gc.collect()

    return _sort_dedupe_cast_klines(out, t_load=t_load, t_merge=t_merge, log=log)


def _utc_seconds(open_time_ms: np.ndarray) -> np.ndarray:
    return (open_time_ms // 1000).astype(np.int64)


def find_valid_window_starts(
    ts_sec: np.ndarray,
    window_size: int,
    lookback: int,
    align_utc_5m: bool,
    require_contiguous: bool,
) -> np.ndarray:
    """
    Return start indices `ws` such that:
    - ws >= lookback (enough pre-window history in the array)
    - ws + window_size <= n
    - optional: ts_sec[ws] % 300 == 0
    - optional: ts_sec[ws:ws+window_size] == ts_sec[ws] + arange(window_size)

    Uses O(n) RAM only. The old sliding_window_view(ts_sec, window_size) path materialized
    an (n × window_size) int64 array and OOMs on tens of millions of rows.
    """
    n = ts_sec.shape[0]
    if n < window_size + lookback:
        return np.array([], dtype=np.int64)

    m = n - window_size + 1
    starts = np.arange(m, dtype=np.int64)

    if require_contiguous:
        # Contiguous 1 Hz: all diffs inside the window must be +1.
        # sum(ok[i:i+L]) == L for L = window_size - 1, ok = (diff(ts)==1).
        ok = np.diff(ts_sec) == 1
        L = window_size - 1
        cs = np.empty(n, dtype=np.int64)
        cs[0] = 0
        cs[1:] = np.cumsum(ok.astype(np.int64, copy=False))
        sums = cs[L:n] - cs[0:m]
        mask = sums == L
    else:
        mask = np.ones(m, dtype=bool)

    if align_utc_5m:
        mask &= (ts_sec[starts] % 300) == 0

    mask &= starts >= lookback
    return starts[mask]


@dataclass
class DatasetBundle:
    """In-memory training tensors + lineage."""

    X: np.ndarray
    y: np.ndarray
    window_id: np.ndarray
    t_index: np.ndarray
    global_idx: np.ndarray
    ts_close_ms: np.ndarray
    feature_names: List[str]
    close: np.ndarray
    open_time_ms: np.ndarray


@dataclass
class PreparedTrainingData:
    """Indicators + window starts + pre-window stats — no stacked X/y (for chunked training)."""

    close: np.ndarray
    open_time_ms: np.ndarray
    window_starts: np.ndarray
    indicators: Dict[str, np.ndarray]
    pre_ret_arr: np.ndarray
    pre_rv_arr: np.ndarray
    feature_names: List[str]
    window_size: int


def _window_price_ok(close: np.ndarray, ws: int, window_size: int) -> bool:
    o = close[ws]
    c_end = close[ws + window_size - 1]
    return bool(np.isfinite(o) and o != 0 and np.isfinite(c_end))


def materialize_window_id_range(
    prep: PreparedTrainingData,
    fc: FeatureConfig,
    wid_lo: int,
    wid_hi: int,
    *,
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
    progress_every_windows: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack rows for window indices ``wid_lo .. wid_hi-1`` (indices into ``prep.window_starts``).
    Skips windows with invalid open/close like ``build_dataset``.
    """
    from config import FEATURE_COLUMNS

    dc_ws = prep.window_size
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    wids: List[np.ndarray] = []
    tis: List[np.ndarray] = []
    gidx: List[np.ndarray] = []
    ts_close: List[np.ndarray] = []
    n_feat = None
    n_ws = wid_hi - wid_lo
    for j, wid in enumerate(range(wid_lo, wid_hi)):
        if verbose and progress_every_windows > 0 and j > 0 and j % progress_every_windows == 0:
            if log:
                log(f"  materialize wid {wid_lo + j:,} / range {wid_lo}-{wid_hi} …")
        ws = int(prep.window_starts[wid])
        if not _window_price_ok(prep.close, ws, dc_ws):
            continue
        o = prep.close[ws]
        c_end = prep.close[ws + dc_ws - 1]
        y_win = float((c_end - o) / o * 100.0)
        xb = vectorized_window_feature_block(
            prep.close,
            prep.indicators,
            ws=ws,
            window_size=dc_ws,
            pre_ret_ws=float(prep.pre_ret_arr[wid]),
            pre_rv_ws=float(prep.pre_rv_arr[wid]),
        )
        if n_feat is None:
            n_feat = xb.shape[1]
        elif xb.shape[1] != n_feat:
            raise RuntimeError("Inconsistent feature width")
        xs.append(xb)
        ys.append(np.full(dc_ws, y_win, dtype=np.float64))
        wids.append(np.full(dc_ws, wid, dtype=np.int64))
        tis.append(np.arange(1, dc_ws + 1, dtype=np.int64))
        gidx.append(np.arange(ws, ws + dc_ws, dtype=np.int64))
        ts_close.append(prep.open_time_ms[ws : ws + dc_ws])

    if not xs:
        nf = len(FEATURE_COLUMNS)
        return (
            np.empty((0, nf), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    X = np.vstack(xs)
    y = np.concatenate(ys)
    window_id = np.concatenate(wids)
    t_index = np.concatenate(tis)
    global_idx = np.concatenate(gidx)
    ts_close_ms = np.concatenate(ts_close)
    return X, y, window_id, t_index, global_idx, ts_close_ms


def materialize_window_id_list(
    prep: PreparedTrainingData,
    fc: FeatureConfig,
    wids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack rows for an arbitrary set of window ids (sorted ascending for determinism)."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    wids_out: List[np.ndarray] = []
    tis: List[np.ndarray] = []
    gidx: List[np.ndarray] = []
    ts_close: List[np.ndarray] = []
    for wid in np.sort(wids):
        X, y, wi, ti, gi, ts = materialize_window_id_range(prep, fc, int(wid), int(wid) + 1)
        if X.shape[0] == 0:
            continue
        xs.append(X)
        ys.append(y)
        wids_out.append(wi)
        tis.append(ti)
        gidx.append(gi)
        ts_close.append(ts)
    if not xs:
        return materialize_window_id_range(prep, fc, 0, 0)
    return (
        np.vstack(xs),
        np.concatenate(ys),
        np.concatenate(wids_out),
        np.concatenate(tis),
        np.concatenate(gidx),
        np.concatenate(ts_close),
    )


def count_valid_rows_for_window_ids(prep: PreparedTrainingData, wids: Union[Sequence[int], np.ndarray]) -> int:
    """Row count if we materialized all listed windows (valid price only)."""
    n = 0
    ws = prep.window_size
    for wid in wids:
        w = int(prep.window_starts[int(wid)])
        if _window_price_ok(prep.close, w, ws):
            n += ws
    return n


def global_wid_row_offsets(prep: PreparedTrainingData) -> Tuple[Dict[int, int], int]:
    """Map window id -> start row in the full stacked dataset (valid windows only)."""
    off: Dict[int, int] = {}
    pos = 0
    ws = prep.window_size
    for wid in range(len(prep.window_starts)):
        w = int(prep.window_starts[wid])
        if _window_price_ok(prep.close, w, ws):
            off[wid] = pos
            pos += ws
    return off, pos


def inner_wid_row_offsets(prep: PreparedTrainingData, inner_wids: np.ndarray) -> Tuple[Dict[int, int], int]:
    """
    Map window id -> start row index in a flat inner-only matrix (valid windows only),
    in ascending ``wid`` order (matches ``build_dataset`` / ``materialize`` ordering).
    """
    off: Dict[int, int] = {}
    pos = 0
    ws = prep.window_size
    for wid in np.sort(inner_wids):
        w = int(prep.window_starts[int(wid)])
        if _window_price_ok(prep.close, w, ws):
            off[int(wid)] = pos
            pos += ws
    return off, pos


def prepare_training_data(
    cfg: Config,
    df: Optional[pd.DataFrame] = None,
    *,
    verbose: bool = False,
) -> PreparedTrainingData:
    """
    Load CSVs, compute indicators and pre-window stats — **does not** stack X/y (saves RAM).
    """
    from config import FEATURE_COLUMNS

    dc = cfg.data
    fc = cfg.features
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"[dataset +{elapsed:8.1f}s] {msg}", flush=True)

    if df is None:
        _log(f"Loading CSVs from {cfg.paths.data_dir} ({dc.csv_glob}) …")
        df = load_binance_1s_csvs(
            cfg.paths.data_dir,
            dc.csv_glob,
            log=_log if verbose else None,
            read_wave_size=dc.csv_read_wave_size,
            merge_batch_size=dc.csv_merge_batch_size,
            use_disk_spill=dc.csv_disk_spill_merge,
            parquet_merge_strategy=dc.csv_parquet_merge_strategy,
        )
        _log(f"Ready: {len(df):,} 1s rows after load.")

    close = df["close"].to_numpy(dtype=np.float32)
    open_time_ms = df["open_time"].to_numpy(dtype=np.int64)
    del df
    gc.collect()
    ts_sec = _utc_seconds(open_time_ms)

    _log("Finding valid 5m windows …")
    window_starts = find_valid_window_starts(
        ts_sec,
        window_size=dc.window_size,
        lookback=dc.lookback_context,
        align_utc_5m=dc.align_windows_to_utc_5m,
        require_contiguous=dc.require_contiguous_seconds,
    )

    if window_starts.size == 0:
        raise RuntimeError("No valid windows found; check data coverage and alignment settings.")

    chunk = int(dc.indicator_chunk_rows)
    _log(
        f"Found {window_starts.size:,} window starts. Precomputing indicators on {len(close):,} closes "
        f"(float32; chunk={chunk if chunk > 0 else 'off'}) …"
    )
    t_ind = time.perf_counter()
    if chunk > 0 and len(close) > chunk:
        indicators = precompute_indicators_chunked(close, fc, chunk_size=chunk, dtype=np.float32)
    else:
        indicators = precompute_indicators(close, fc, dtype=np.float32)
    _log(f"Indicators done in {time.perf_counter() - t_ind:.1f}s. Pre-window stats …")
    pre_ret_arr, pre_rv_arr = compute_pre_window_stats(close, window_starts, dc.lookback_context)
    _log(f"Prepared training tensors (no X/y stack) in {time.perf_counter() - t0:.1f}s.")

    return PreparedTrainingData(
        close=close,
        open_time_ms=open_time_ms,
        window_starts=window_starts,
        indicators=indicators,
        pre_ret_arr=pre_ret_arr,
        pre_rv_arr=pre_rv_arr,
        feature_names=list(FEATURE_COLUMNS),
        window_size=dc.window_size,
    )


def build_dataset(
    cfg: Config,
    df: Optional[pd.DataFrame] = None,
    *,
    verbose: bool = False,
    progress_every_windows: int = 500,
) -> DatasetBundle:
    """
    Full pipeline from raw dataframe (or load from disk) to stacked samples.
    Each row: one second inside a 5m window. Target is the window's terminal return (%).

    If ``verbose``, print timing and periodic progress while stacking windows (large data).
    """
    from config import FEATURE_COLUMNS

    dc = cfg.data
    fc = cfg.features
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"[dataset +{elapsed:8.1f}s] {msg}", flush=True)

    prep = prepare_training_data(cfg, df=df, verbose=verbose)
    _log(f"Stacking feature rows ({prep.window_starts.size} windows × {dc.window_size} s) …")
    t_stack = time.perf_counter()
    X, y, window_id, t_index, global_idx, ts_close_ms = materialize_window_id_range(
        prep,
        fc,
        0,
        len(prep.window_starts),
        verbose=verbose,
        log=_log if verbose else None,
        progress_every_windows=progress_every_windows,
    )
    if X.shape[0] == 0:
        raise RuntimeError("All windows skipped (invalid prices).")

    if len(FEATURE_COLUMNS) != X.shape[1]:
        raise RuntimeError("FEATURE_COLUMNS length must match feature matrix width.")

    nbytes = X.nbytes + y.nbytes
    _log(
        f"Done stacking in {time.perf_counter() - t_stack:.1f}s / {time.perf_counter() - t0:.1f}s total. "
        f"X={tuple(X.shape)} ~{nbytes / (1024 ** 3):.2f} GiB ({X.dtype})."
    )

    return DatasetBundle(
        X=X,
        y=y,
        window_id=window_id,
        t_index=t_index,
        global_idx=global_idx,
        ts_close_ms=ts_close_ms,
        feature_names=list(FEATURE_COLUMNS),
        close=prep.close,
        open_time_ms=prep.open_time_ms,
    )


def iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield random minibatches (optional helper for torch)."""
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        j = idx[s:e]
        yield X[j], y[j]
