"""
Pre-compute rolling / contextual indicators on the full 1-second close series,
then slice per (window, intra-second) row without leakage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config import FeatureConfig


def indicator_lookback_samples(fc: FeatureConfig) -> int:
    """How many trailing samples indicators may read before index i (warmup / ROC lag)."""
    return int(
        max(
            max(fc.roc_horizons),
            max(fc.ctx_roc_horizons),
            max(fc.zscore_windows),
            fc.bollinger_period,
            fc.rsi_period,
            fc.ema_period,
        )
    )


def _indicator_output_keys(fc: FeatureConfig) -> list[str]:
    keys: list[str] = []
    for lag in fc.roc_horizons:
        keys.append(f"roc_{lag}s")
    for w in fc.zscore_windows:
        keys.append(f"zscore_{w}")
    keys.extend(["bb_pct_b", "rsi_14", "dist_ema_60"])
    for lag in fc.ctx_roc_horizons:
        keys.append(f"ctx_roc_{lag}")
    return keys


def _safe_div(num: np.ndarray, den: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=dtype)
    m = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[m] = (num[m] / den[m]).astype(dtype, copy=False)
    return out


def rsi_wilder(close: np.ndarray, period: int = 14, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """RSI (Wilder) on 1s closes; returns NaN until warmed."""
    s = pd.Series(close, dtype=dtype)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.to_numpy(dtype=dtype)


def ema(close: np.ndarray, span: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    s = pd.Series(close, dtype=dtype)
    return s.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy(dtype=dtype)


def bollinger_pct_b(
    close: np.ndarray, period: int, num_std: float, *, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns middle, upper, lower, pct_b."""
    s = pd.Series(close, dtype=dtype)
    mid = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (s - lower) / width
    return (
        mid.to_numpy(dtype=dtype),
        upper.to_numpy(dtype=dtype),
        lower.to_numpy(dtype=dtype),
        pct_b.to_numpy(dtype=dtype),
    )


def rolling_zscore(
    close: np.ndarray, window: int, *, dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close, dtype=dtype)
    mu = s.rolling(window=window, min_periods=window).mean()
    sig = s.rolling(window=window, min_periods=window).std(ddof=0)
    z = (s - mu) / sig.replace(0.0, np.nan)
    return mu.to_numpy(dtype=dtype), sig.to_numpy(dtype=dtype), z.to_numpy(dtype=dtype)


def pct_roc(close: np.ndarray, lag: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Percent change over `lag` seconds: (c[t]-c[t-lag])/c[t-lag]*100."""
    n = len(close)
    out = np.full(n, np.nan, dtype=dtype)
    if lag <= 0 or n <= lag:
        return out
    prev = close[:-lag]
    cur = close[lag:]
    num = (cur - prev) * np.asarray(100.0, dtype=dtype)
    out[lag:] = _safe_div(num, prev, dtype=dtype)
    return out


def precompute_indicators(
    close: np.ndarray,
    fc: FeatureConfig,
    *,
    dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """
    Vectorized indicators aligned index-wise with `close`.
    All arrays length len(close). Default ``dtype`` is float32 to halve RAM vs float64.
    """
    close = np.asarray(close, dtype=dtype, order="C")
    out: dict[str, np.ndarray] = {}

    for lag in fc.roc_horizons:
        out[f"roc_{lag}s"] = pct_roc(close, lag, dtype=dtype)

    for w in fc.zscore_windows:
        _, _, z = rolling_zscore(close, w, dtype=dtype)
        out[f"zscore_{w}"] = z

    _, _, _, out["bb_pct_b"] = bollinger_pct_b(close, fc.bollinger_period, fc.bollinger_num_std, dtype=dtype)
    out["rsi_14"] = rsi_wilder(close, fc.rsi_period, dtype=dtype)
    ema_v = ema(close, fc.ema_period, dtype=dtype)
    scale = np.asarray(100.0, dtype=dtype)
    out["dist_ema_60"] = (_safe_div(close - ema_v, close, dtype=dtype) * scale).astype(dtype, copy=False)

    for lag in fc.ctx_roc_horizons:
        out[f"ctx_roc_{lag}"] = pct_roc(close, lag, dtype=dtype)

    return out


def precompute_indicators_chunked(
    close: np.ndarray,
    fc: FeatureConfig,
    *,
    chunk_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """
    Same keys/values as ``precompute_indicators``, but process ``close`` in contiguous
    chunks with a trailing warmup slice so rolling/ROC logic matches a full pass.

    Peak RAM ~ O(chunk_size * n_indicators), not O(len(close) * n_indicators).
    """
    n = int(close.shape[0])
    if n == 0:
        return {}
    chunk_size = max(1, int(chunk_size))
    overlap = indicator_lookback_samples(fc) + 2
    keys = _indicator_output_keys(fc)
    out = {k: np.empty(n, dtype=dtype) for k in keys}

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        s = max(0, start - overlap)
        chunk_close = np.asarray(close[s:end], dtype=dtype, order="C")
        part = precompute_indicators(chunk_close, fc, dtype=dtype)
        off = start - s
        length = end - start
        for k in keys:
            out[k][start:end] = part[k][off : off + length]

    return out


def compute_pre_window_stats(close: np.ndarray, window_starts: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per window start index `ws`: return over [ws-lookback, ws) and realized vol of log returns in that interval.
    """
    n_win = len(window_starts)
    pre_ret = np.full(n_win, np.nan, dtype=np.float64)
    pre_rv = np.full(n_win, np.nan, dtype=np.float64)
    log_close = np.log(np.clip(close, 1e-12, np.inf))

    for k, ws in enumerate(window_starts):
        a = int(ws - lookback)
        b = int(ws)
        if a < 0 or b <= a + 1:
            continue
        c0 = close[a]
        c1 = close[b - 1]
        if not np.isfinite(c0) or c0 == 0:
            continue
        pre_ret[k] = (c1 - c0) / c0 * 100.0
        lr = np.diff(log_close[a:b])
        if lr.size > 1:
            pre_rv[k] = float(np.std(lr, ddof=0))

    return pre_ret, pre_rv


def intra_window_cum_log_returns(close: np.ndarray, ws: int, window_size: int) -> np.ndarray:
    """
    For each intra-window second t=1..window_size (index offset 0..window_size-1),
    cumulative sum of 1s log returns from window open through current second.
    """
    fd = np.float32 if getattr(close, "dtype", np.float64) == np.float32 else np.float64
    seg = close[ws : ws + window_size]
    if seg.size < window_size:
        return np.full(window_size, np.nan, dtype=fd)
    lr = np.log(np.clip(seg[1:], 1e-12, np.inf) / np.clip(seg[:-1], 1e-12, np.inf))
    z = np.array(0.0, dtype=fd)
    cum = np.concatenate([z.reshape(1), np.cumsum(lr.astype(fd, copy=False))])
    return cum.astype(fd, copy=False)


def vectorized_window_feature_block(
    close: np.ndarray,
    indicators: dict[str, np.ndarray],
    ws: int,
    window_size: int,
    pre_ret_ws: float,
    pre_rv_ws: float,
) -> np.ndarray:
    """
    Build X of shape (window_size, n_features) for one window starting at `ws`.
    Row k corresponds to t = k+1, global index idx = ws + k.
    """
    fd = np.float32 if close.dtype == np.float32 else np.float64
    idxs = ws + np.arange(window_size, dtype=np.int64)
    c_ws = close[ws]
    t_idx = np.arange(1, window_size + 1, dtype=fd)
    time_left = (np.float32(window_size) - t_idx).astype(fd) if fd == np.float32 else (window_size - t_idx)

    if not np.isfinite(c_ws) or c_ws == 0:
        pct_so_far = np.full(window_size, np.nan, dtype=fd)
    else:
        scale = np.asarray(100.0, dtype=fd)
        pct_so_far = ((close[idxs] - c_ws) / c_ws * scale).astype(fd, copy=False)

    cum_log = intra_window_cum_log_returns(close, ws, window_size)

    cols = [
        time_left,
        t_idx,
        pct_so_far,
        cum_log,
        indicators["roc_5s"][idxs],
        indicators["roc_15s"][idxs],
        indicators["roc_30s"][idxs],
        indicators["roc_60s"][idxs],
        indicators["zscore_60"][idxs],
        indicators["zscore_300"][idxs],
        indicators["bb_pct_b"][idxs],
        indicators["rsi_14"][idxs],
        indicators["dist_ema_60"][idxs],
        indicators["ctx_roc_300"][idxs],
        indicators["ctx_roc_600"][idxs],
        np.full(window_size, pre_ret_ws, dtype=fd),
        np.full(window_size, pre_rv_ws, dtype=fd),
    ]
    return np.column_stack(cols).astype(fd, copy=False)
