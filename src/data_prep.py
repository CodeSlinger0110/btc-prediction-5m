"""
Load Binance 1s klines, discover valid 5-minute windows, and materialize (X, y, meta).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
from features import compute_pre_window_stats, precompute_indicators, vectorized_window_feature_block


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


def load_binance_1s_csvs(data_dir: Path, glob_pattern: str) -> pd.DataFrame:
    """Load and concatenate all matching CSVs, sorted by open_time."""
    paths = sorted(data_dir.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {glob_pattern} under {data_dir}")

    chunks: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, header=None, names=BINANCE_COLS[:12])
        chunks.append(df)

    out = pd.concat(chunks, ignore_index=True)
    out["open_time"] = out["open_time"].astype(np.int64)
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    out = out.reset_index(drop=True)

    for c in ("open", "high", "low", "close", "volume"):
        out[c] = out[c].astype(np.float64)

    return out


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
    """
    n = ts_sec.shape[0]
    if n < window_size + lookback:
        return np.array([], dtype=np.int64)

    from numpy.lib.stride_tricks import sliding_window_view

    w = sliding_window_view(ts_sec, window_size)
    expected = w[:, 0:1] + np.arange(window_size, dtype=np.int64)
    if require_contiguous:
        valid_time = np.all(w == expected, axis=1)
    else:
        valid_time = np.ones(w.shape[0], dtype=bool)
    mask = valid_time
    if align_utc_5m:
        mask &= (w[:, 0] % 300) == 0
    starts = np.arange(n - window_size + 1, dtype=np.int64)
    idx = starts[mask]
    idx = idx[idx >= lookback]
    return idx.astype(np.int64)


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
    dc = cfg.data
    fc = cfg.features
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"[dataset +{elapsed:8.1f}s] {msg}", flush=True)

    if df is None:
        _log(f"Loading CSVs from {cfg.paths.data_dir} ({dc.csv_glob}) …")
        df = load_binance_1s_csvs(cfg.paths.data_dir, dc.csv_glob)
        _log(f"Loaded {len(df):,} 1s rows.")

    close = df["close"].to_numpy(dtype=np.float64)
    open_time_ms = df["open_time"].to_numpy(dtype=np.int64)
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

    _log(f"Found {window_starts.size:,} window starts. Precomputing indicators on {len(close):,} closes …")
    t_ind = time.perf_counter()
    indicators = precompute_indicators(close, fc)
    _log(f"Indicators done in {time.perf_counter() - t_ind:.1f}s. Pre-window stats …")
    pre_ret_arr, pre_rv_arr = compute_pre_window_stats(close, window_starts, dc.lookback_context)
    _log(f"Stacking feature rows ({window_starts.size} windows × {dc.window_size} s) …")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    wids: List[np.ndarray] = []
    tis: List[np.ndarray] = []
    gidx: List[np.ndarray] = []
    ts_close: List[np.ndarray] = []

    n_feat = None
    n_ws = int(window_starts.size)
    for wid, ws in enumerate(window_starts):
        if verbose and progress_every_windows > 0 and wid > 0 and wid % progress_every_windows == 0:
            _log(f"  windows {wid:,} / {n_ws:,} ({100.0 * wid / n_ws:.0f}%) …")
        w = int(ws)
        o = close[w]
        c_end = close[w + dc.window_size - 1]
        if not np.isfinite(o) or o == 0 or not np.isfinite(c_end):
            continue
        y_win = (c_end - o) / o * 100.0

        xb = vectorized_window_feature_block(
            close,
            indicators,
            ws=w,
            window_size=dc.window_size,
            pre_ret_ws=float(pre_ret_arr[wid]),
            pre_rv_ws=float(pre_rv_arr[wid]),
        )
        if n_feat is None:
            n_feat = xb.shape[1]
        elif xb.shape[1] != n_feat:
            raise RuntimeError("Inconsistent feature width")

        xs.append(xb)
        ys.append(np.full(dc.window_size, y_win, dtype=np.float64))
        wids.append(np.full(dc.window_size, wid, dtype=np.int64))
        tis.append(np.arange(1, dc.window_size + 1, dtype=np.int64))
        gidx.append(np.arange(w, w + dc.window_size, dtype=np.int64))
        ts_close.append(open_time_ms[w : w + dc.window_size])

    if not xs:
        raise RuntimeError("All windows skipped (invalid prices).")

    _log("Concatenating arrays …")
    t_cat = time.perf_counter()
    X = np.vstack(xs)
    y = np.concatenate(ys)
    window_id = np.concatenate(wids)
    t_index = np.concatenate(tis)
    global_idx = np.concatenate(gidx)
    ts_close_ms = np.concatenate(ts_close)

    from config import FEATURE_COLUMNS

    if len(FEATURE_COLUMNS) != X.shape[1]:
        raise RuntimeError("FEATURE_COLUMNS length must match feature matrix width.")

    nbytes = X.nbytes + y.nbytes
    _log(
        f"Done in {time.perf_counter() - t_cat:.1f}s concat / {time.perf_counter() - t0:.1f}s total. "
        f"X={tuple(X.shape)} ~{nbytes / (1024 ** 3):.2f} GiB float64."
    )

    return DatasetBundle(
        X=X,
        y=y,
        window_id=window_id,
        t_index=t_index,
        global_idx=global_idx,
        ts_close_ms=ts_close_ms,
        feature_names=list(FEATURE_COLUMNS),
        close=close,
        open_time_ms=open_time_ms,
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
