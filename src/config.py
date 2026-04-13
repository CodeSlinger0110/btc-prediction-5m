"""
Central configuration for intra-window BTC 5m Polymarket-style modeling.
Edit paths and hyperparameters here or override via environment / CLI in train.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PathsConfig:
    """Filesystem layout."""

    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.repo_root / "data"
        self.models_dir = self.repo_root / "models"
        self.outputs_dir = self.repo_root / "models" / "artifacts"


@dataclass
class DataConfig:
    """Raw data and windowing."""

    # Glob under data_dir for Binance 1s klines (no header).
    csv_glob: str = "BTCUSDT-1s-*.csv"
    # Merge this many CSV DataFrames into one block before writing a shard / in-memory block (1 = never concat CSVs in RAM).
    csv_read_wave_size: int = 1
    # Fan-in per level when tree-merging shards/blocks (2 = pairwise merge; lower peak RAM than 4+).
    # Used only when csv_parquet_merge_strategy is "tree" (disk spill).
    csv_merge_batch_size: int = 2
    # Disk spill: "sequential" = append one parquet shard at a time (lowest peak RAM).
    # "tree" = pairwise PyArrow concat (faster on fast disks, higher peak RAM).
    csv_parquet_merge_strategy: str = "sequential"
    # When pyarrow is installed, write wave-blocks to temp parquet and merge on disk (avoids OOM on many large files).
    csv_disk_spill_merge: bool = True
    # Seconds per prediction window (5-minute Polymarket candle).
    window_size: int = 300
    # Seconds of 1s history required *before* window open (for pre-window stats + warm rolls).
    lookback_context: int = 600
    # Only use windows whose first timestamp is aligned to UTC 5-minute boundaries.
    align_windows_to_utc_5m: bool = True
    # Require 300 consecutive 1-second timestamps inside each window.
    require_contiguous_seconds: bool = True
    # Precompute rolling indicators in contiguous slices of this many rows (plus a warmup tail).
    # 0 = one pass (highest peak RAM). ~2e6 keeps peak indicator memory low on 50M+ row series.
    indicator_chunk_rows: int = 2_000_000


@dataclass
class FeatureConfig:
    """Feature engineering knobs."""

    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_num_std: float = 2.0
    ema_period: int = 60
    zscore_windows: tuple[int, ...] = (60, 300)
    roc_horizons: tuple[int, ...] = (5, 15, 30, 60)
    # Extended momentum horizons reaching into lookback_context.
    ctx_roc_horizons: tuple[int, ...] = (300, 600)


@dataclass
class TrainConfig:
    """Training, validation, and backtest."""

    model_type: str = "lightgbm"  # "lightgbm" | "lstm"
    random_seed: int = 42
    # Walk-forward: number of splits over ordered windows (TimeSeriesSplit style).
    walk_forward_splits: int = 5
    # Minimum windows per train fold (expanding; first fold uses at least this many).
    min_train_windows: int = 200
    # Fraction of chronologically last windows held out for final reporting only.
    final_holdout_fraction: float = 0.08
    # LightGBM
    lgb_num_leaves: int = 63
    lgb_learning_rate: float = 0.05
    lgb_n_estimators: int = 800
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_early_stopping_rounds: int = 50
    lgb_verbose: int = -1
    # LSTM (optional)
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_dropout: float = 0.1
    lstm_seq_len: int = 64
    lstm_epochs: int = 15
    lstm_batch_size: int = 512
    lstm_lr: float = 1e-3
    # Simulated trade: use prediction from this intra-window second (1..300) for directional PnL.
    backtest_decision_t: int = 300
    # Annualization for Sharpe on 5m windows (24/7): 12 * 24 * 365.
    periods_per_year: float = 12.0 * 24.0 * 365.0
    # LightGBM: max training windows per RAM batch (0 = build one giant X/y — needs huge RAM).
    # Use 500–2000 on long histories so train.py materializes/scales/boosts in chunks.
    train_chunk_windows: int = 0


@dataclass
class Config:
    """Top-level bundle."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def ensure_dirs(self) -> None:
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)


# Stable feature column order (must match features.build_feature_matrix).
FEATURE_COLUMNS: List[str] = [
    "time_left",
    "t_index",
    "pct_change_so_far",
    "cum_log_ret_window",
    "roc_5s",
    "roc_15s",
    "roc_30s",
    "roc_60s",
    "zscore_60",
    "zscore_300",
    "bb_pct_b",
    "rsi_14",
    "dist_ema_60",
    "ctx_roc_300",
    "ctx_roc_600",
    "pre_window_ret",
    "pre_window_realized_vol",
]
