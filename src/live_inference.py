"""
Build a single feature row from a 900-second synthetic series (600 pre-window + 300 window)
for live streaming ticks. Matches training feature order in config.FEATURE_COLUMNS.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import Config, FeatureConfig
from features import precompute_indicators


def remain_sec_to_t_index(remain_sec: int, window_size: int = 300) -> int:
    """Maps Polymarket-style seconds remaining to intra-window t (1..window_size)."""
    r = int(round(remain_sec))
    r = max(1, min(window_size, r))
    return max(1, min(window_size, 301 - r))


def pred_to_up_down_pct(pred_pct: float, scale: float = 18.0) -> Tuple[float, float]:
    """
    Map predicted final-window return (percent points) to implied Up / Down settlement weights.
    Uses a logistic squish; scale tunes sensitivity (~0.05% move -> moderate tilt).
    """
    if not math.isfinite(pred_pct):
        return 50.0, 50.0
    p_up = 100.0 / (1.0 + math.exp(-float(pred_pct) * scale))
    p_up = max(1.0, min(99.0, p_up))
    return p_up, 100.0 - p_up


def _fill_nan_1d(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a, dtype="float64")
    return s.ffill().bfill().to_numpy(dtype=np.float64)


def build_close_array_900(
    bucket_start_sec: int,
    window_open: float,
    sec_to_close: Dict[int, float],
    lookback: int = 600,
    window_size: int = 300,
) -> np.ndarray:
    """
    Indices 0..599 = pre-window [bucket_start-600, bucket_start-1],
    indices 600..899 = window [bucket_start, bucket_start+299].
    Missing seconds filled forward/backward after seeding with window_open.
    """
    out = np.full(lookback + window_size, np.nan, dtype=np.float64)
    ws = lookback
    for i in range(lookback):
        abs_sec = bucket_start_sec - lookback + i
        if abs_sec in sec_to_close:
            out[i] = sec_to_close[abs_sec]
    for j in range(window_size):
        abs_sec = bucket_start_sec + j
        if abs_sec in sec_to_close:
            out[ws + j] = sec_to_close[abs_sec]
    if math.isfinite(window_open) and window_open > 0:
        base = float(window_open)
        s = pd.Series(out, dtype="float64")
        s = s.fillna(base)
        out = s.to_numpy(dtype=np.float64)
    out = _fill_nan_1d(out)
    return out


def extract_single_row(
    close: np.ndarray,
    ws: int,
    t_index: int,
    fc: FeatureConfig,
) -> np.ndarray:
    """One row of features at global index ws + t_index - 1 (training-aligned)."""
    window_size = 300
    if close.shape[0] < ws + window_size:
        raise ValueError("close array too short")
    indicators = precompute_indicators(close, fc)
    idx = ws + t_index - 1
    c_ws = close[ws]
    c_i = close[idx]
    if not np.isfinite(c_ws) or c_ws == 0:
        pct_so_far = np.nan
        cum_log = np.nan
    else:
        pct_so_far = (c_i - c_ws) / c_ws * 100.0
        seg = close[ws : idx + 1]
        if len(seg) >= 2:
            cum_log = float(np.sum(np.log(np.clip(seg[1:], 1e-12, np.inf) / np.clip(seg[:-1], 1e-12, np.inf))))
        else:
            cum_log = 0.0

    time_left = float(window_size - t_index)
    t_idx = float(t_index)

    pre_seg = close[0:ws]
    if pre_seg.size > 1 and np.isfinite(pre_seg[0]) and pre_seg[0] > 0:
        pre_ret = (pre_seg[-1] - pre_seg[0]) / pre_seg[0] * 100.0
        lr = np.diff(np.log(np.clip(pre_seg, 1e-12, np.inf)))
        pre_rv = float(np.std(lr, ddof=0)) if lr.size > 1 else 0.0
    else:
        pre_ret = 0.0
        pre_rv = 0.0

    row = np.array(
        [
            time_left,
            t_idx,
            float(pct_so_far),
            float(cum_log),
            float(indicators["roc_5s"][idx]),
            float(indicators["roc_15s"][idx]),
            float(indicators["roc_30s"][idx]),
            float(indicators["roc_60s"][idx]),
            float(indicators["zscore_60"][idx]),
            float(indicators["zscore_300"][idx]),
            float(indicators["bb_pct_b"][idx]),
            float(indicators["rsi_14"][idx]),
            float(indicators["dist_ema_60"][idx]),
            float(indicators["ctx_roc_300"][idx]),
            float(indicators["ctx_roc_600"][idx]),
            float(pre_ret),
            float(pre_rv),
        ],
        dtype=np.float64,
    )
    return row


class LiveMarketBuffer:
    """
    Per-market 1 Hz closes keyed by Unix second, plus bucket metadata for feature construction.
    """

    def __init__(self, market_id: str, bucket_start_sec: int, window_open: float) -> None:
        self.market_id = market_id
        self.bucket_start_sec = int(bucket_start_sec)
        self.window_open = float(window_open)
        self.sec_to_close: Dict[int, float] = {}

    def add_second(self, unix_sec: int, price: float) -> None:
        if np.isfinite(price) and price > 0:
            self.sec_to_close[int(unix_sec)] = float(price)

    def feature_row(
        self,
        remain_sec: int,
        cfg: Config,
    ) -> np.ndarray:
        t_index = remain_sec_to_t_index(remain_sec, cfg.data.window_size)
        close = build_close_array_900(
            self.bucket_start_sec,
            self.window_open,
            self.sec_to_close,
            lookback=cfg.data.lookback_context,
            window_size=cfg.data.window_size,
        )
        ws = cfg.data.lookback_context
        return extract_single_row(close, ws, t_index, cfg.features)
