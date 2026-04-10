"""
Pre-compute rolling / contextual indicators on the full 1-second close series,
then slice per (window, intra-second) row without leakage.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config import FeatureConfig


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, dtype=np.float64, fill_value=np.nan)
    m = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[m] = num[m] / den[m]
    return out


def rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Wilder) on 1s closes; returns NaN until warmed."""
    s = pd.Series(close, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.to_numpy(dtype=np.float64)


def ema(close: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(close, dtype="float64")
    return s.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy(dtype=np.float64)


def bollinger_pct_b(close: np.ndarray, period: int, num_std: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns middle, upper, lower, pct_b."""
    s = pd.Series(close, dtype="float64")
    mid = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (s - lower) / width
    return (
        mid.to_numpy(dtype=np.float64),
        upper.to_numpy(dtype=np.float64),
        lower.to_numpy(dtype=np.float64),
        pct_b.to_numpy(dtype=np.float64),
    )


def rolling_zscore(close: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close, dtype="float64")
    mu = s.rolling(window=window, min_periods=window).mean()
    sig = s.rolling(window=window, min_periods=window).std(ddof=0)
    z = (s - mu) / sig.replace(0.0, np.nan)
    return mu.to_numpy(dtype=np.float64), sig.to_numpy(dtype=np.float64), z.to_numpy(dtype=np.float64)


def pct_roc(close: np.ndarray, lag: int) -> np.ndarray:
    """Percent change over `lag` seconds: (c[t]-c[t-lag])/c[t-lag]*100."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if lag <= 0 or n <= lag:
        return out
    prev = close[:-lag]
    cur = close[lag:]
    out[lag:] = _safe_div((cur - prev) * 100.0, prev)
    return out


def precompute_indicators(close: np.ndarray, fc: FeatureConfig) -> dict[str, np.ndarray]:
    """
    Vectorized indicators aligned index-wise with `close`.
    All arrays length len(close).
    """
    out: dict[str, np.ndarray] = {}

    for lag in fc.roc_horizons:
        out[f"roc_{lag}s"] = pct_roc(close, lag)

    for w in fc.zscore_windows:
        _, _, z = rolling_zscore(close, w)
        out[f"zscore_{w}"] = z

    _, _, _, out["bb_pct_b"] = bollinger_pct_b(close, fc.bollinger_period, fc.bollinger_num_std)
    out["rsi_14"] = rsi_wilder(close, fc.rsi_period)
    ema_v = ema(close, fc.ema_period)
    out["dist_ema_60"] = _safe_div(close - ema_v, close) * 100.0

    for lag in fc.ctx_roc_horizons:
        out[f"ctx_roc_{lag}"] = pct_roc(close, lag)

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
    seg = close[ws : ws + window_size]
    if seg.size < window_size:
        return np.full(window_size, np.nan, dtype=np.float64)
    lr = np.log(np.clip(seg[1:], 1e-12, np.inf) / np.clip(seg[:-1], 1e-12, np.inf))
    cum = np.concatenate([[0.0], np.cumsum(lr)])
    return cum.astype(np.float64)


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
    idxs = ws + np.arange(window_size, dtype=np.int64)
    c_ws = close[ws]
    t_idx = np.arange(1, window_size + 1, dtype=np.float64)
    time_left = (window_size - t_idx).astype(np.float64)

    if not np.isfinite(c_ws) or c_ws == 0:
        pct_so_far = np.full(window_size, np.nan, dtype=np.float64)
    else:
        pct_so_far = (close[idxs] - c_ws) / c_ws * 100.0

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
        np.full(window_size, pre_ret_ws, dtype=np.float64),
        np.full(window_size, pre_rv_ws, dtype=np.float64),
    ]
    return np.column_stack(cols)
