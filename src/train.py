"""
Walk-forward training (LightGBM or small LSTM), evaluation, and artifact export.

Run from repo root:
  python src/train.py

Or from src/:
  python train.py

Progress logs (timestamps, dataset stages, each CV fold, LightGBM iteration metrics):
  python src/train.py
Quiet + custom LightGBM print interval:
  python src/train.py --quiet
  python src/train.py --lgb-log-period 100 --dataset-progress-every 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow `python src/train.py` imports
_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config, TrainConfig
from data_prep import DatasetBundle, build_dataset

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# Set by main() so helpers can log without threading state everywhere.
_QUIET: bool = False


def _log(msg: str) -> None:
    if not _QUIET:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _lgb_eval_callbacks(period: int) -> List[Any]:
    """LightGBM iteration logging; empty if period <= 0."""
    if lgb is None or period <= 0:
        return []
    return [lgb.log_evaluation(period=period)]


if nn is not None:

    class LSTMRegressor(nn.Module):
        """Compact univariate-return sequence encoder."""

        def __init__(self, seq_len: int, hidden: int, layers: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            y, _ = self.lstm(x)
            last = y[:, -1, :]
            return self.head(last).squeeze(-1)

else:
    LSTMRegressor = None  # type: ignore[misc, assignment]


def _annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=0))
    if sig == 0.0:
        return float("nan")
    return (mu / sig) * float(np.sqrt(periods_per_year))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.sign(y_true[m]) == np.sign(y_pred[m])))


def simulated_trade_returns(
    y_true_pct: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Long if pred>0 else short; PnL in fractional return space using realized window move (%).
    """
    pos = np.sign(y_pred)
    pos[pos == 0] = 1.0
    return pos * (y_true_pct / 100.0)


def build_return_sequences(close: np.ndarray, global_idx: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Per row, last `seq_len` one-second simple returns ending at `global_idx`.
    Pads with 0 on the left if history is shorter than seq_len.
    """
    n = global_idx.shape[0]
    ret = np.zeros_like(close)
    ret[1:] = (close[1:] - close[:-1]) / np.clip(close[:-1], 1e-12, np.inf)

    out = np.zeros((n, seq_len), dtype=np.float32)
    for i in range(n):
        gi = int(global_idx[i])
        start = gi - seq_len + 1
        if start >= 1:
            chunk = ret[start : gi + 1]
            if chunk.size == seq_len:
                out[i] = chunk.astype(np.float32)
        else:
            # pad early series indices
            avail = ret[1 : gi + 1]
            take = min(seq_len, avail.size)
            if take > 0:
                out[i, seq_len - take :] = avail[-take:].astype(np.float32)
    return out


def train_lstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    tc: TrainConfig,
    device: str,
    *,
    verbose: bool = True,
) -> Tuple[Any, StandardScaler]:
    assert torch is not None and nn is not None and LSTMRegressor is not None
    torch.manual_seed(tc.random_seed)
    np.random.seed(tc.random_seed)

    scaler = StandardScaler()
    flat = X_seq[train_mask].reshape(-1, 1)
    scaler.fit(flat)
    X_tr = scaler.transform(X_seq[train_mask].reshape(-1, 1)).reshape(-1, tc.lstm_seq_len, 1).astype(np.float32)
    X_va = scaler.transform(X_seq[val_mask].reshape(-1, 1)).reshape(-1, tc.lstm_seq_len, 1).astype(np.float32)
    y_tr = y[train_mask].astype(np.float32)
    y_va = y[val_mask].astype(np.float32)

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=tc.lstm_batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=tc.lstm_batch_size, shuffle=False)

    model = LSTMRegressor(tc.lstm_seq_len, tc.lstm_hidden, tc.lstm_layers, tc.lstm_dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lstm_lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    for epoch in range(tc.lstm_epochs):
        t_ep = time.perf_counter()
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for xb, yb in dl_va:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
            pv = np.concatenate(preds) if preds else np.array([])
            mse = float(np.mean((pv - y_va.numpy()) ** 2)) if pv.size else float("inf")
        if verbose:
            _log(
                f"  LSTM epoch {epoch + 1}/{tc.lstm_epochs}  val_mse={mse:.8f}  "
                f"{'*best*' if mse < best_val else ''}  ({time.perf_counter() - t_ep:.1f}s)"
            )
        if mse < best_val:
            best_val = mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def predict_lstm(model: Any, scaler: StandardScaler, X_seq: np.ndarray, device: str) -> np.ndarray:
    assert torch is not None
    model.eval()
    X = scaler.transform(X_seq.reshape(-1, 1)).reshape(-1, X_seq.shape[1], 1).astype(np.float32)
    out: List[np.ndarray] = []
    bs = 4096
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            e = min(s + bs, X.shape[0])
            xb = torch.from_numpy(X[s:e]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def walk_forward_masks(
    unique_window_ids: np.ndarray,
    n_splits: int,
    min_train_windows: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window splits on ordered unique window ids (integers 0..K-1).
    Returns list of (train_window_ids, test_window_ids).
    """
    k = unique_window_ids.size
    if k < min_train_windows + 2:
        raise ValueError(f"Need more windows for walk-forward (have {k}).")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    # Map fold indices to window id sets
    for train_idx, test_idx in tscv.split(unique_window_ids):
        tr_w = unique_window_ids[train_idx]
        te_w = unique_window_ids[test_idx]
        if tr_w.size < min_train_windows:
            continue
        folds.append((tr_w, te_w))
    if not folds:
        # Fallback: single split 70/30 by time
        cut = int(k * 0.7)
        folds.append((unique_window_ids[:cut], unique_window_ids[cut:]))
    return folds


def aggregate_fold_backtest(
    bundle: DatasetBundle,
    y_pred: np.ndarray,
    decision_t: int,
) -> pd.DataFrame:
    """One row per window at decision second."""
    m = (bundle.t_index == decision_t) & np.isfinite(y_pred)
    df = pd.DataFrame(
        {
            "window_id": bundle.window_id[m],
            "t_index": bundle.t_index[m],
            "ts_close_ms": bundle.ts_close_ms[m],
            "y_true_pct": bundle.y[m],
            "y_pred_pct": y_pred[m],
        }
    )
    df["position"] = np.sign(df["y_pred_pct"].to_numpy())
    df.loc[df["position"] == 0, "position"] = 1.0
    df["pnl_frac"] = df["position"] * (df["y_true_pct"] / 100.0)
    return df


def run_walk_forward_lightgbm(
    bundle: DatasetBundle,
    cfg: Config,
    inner_unique_ids: np.ndarray,
    *,
    lgb_log_period: int = 50,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Returns metrics dict and walk-forward test predictions on inner rows (NaN where not in a test fold)."""
    assert lgb is not None
    tc = cfg.train
    X = bundle.X
    y = bundle.y
    wid = bundle.window_id

    inner_mask = np.isin(wid, inner_unique_ids)
    X_inner = X[inner_mask]
    y_inner = y[inner_mask]
    wid_inner = wid[inner_mask]
    t_inner = bundle.t_index[inner_mask]
    g_inner = bundle.global_idx[inner_mask]
    ts_inner = bundle.ts_close_ms[inner_mask]

    unique_sorted = np.sort(np.unique(wid_inner))
    folds = walk_forward_masks(unique_sorted, tc.walk_forward_splits, tc.min_train_windows)
    _log(f"Walk-forward: {len(folds)} fold(s), inner rows={X_inner.shape[0]:,}, features={X_inner.shape[1]}")

    oof_pred = np.full(X_inner.shape[0], np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, Any]] = []

    eval_cb = _lgb_eval_callbacks(lgb_log_period)
    for fold_id, (tr_w, te_w) in enumerate(folds):
        train_m = np.isin(wid_inner, tr_w)
        test_m = np.isin(wid_inner, te_w)
        n_tr = int(train_m.sum())
        n_te = int(test_m.sum())
        _log(
            f"Fold {fold_id + 1}/{len(folds)}: train rows={n_tr:,} ({np.unique(tr_w).size} windows), "
            f"test rows={n_te:,} ({np.unique(te_w).size} windows) — fitting LightGBM …"
        )
        t_fold = time.perf_counter()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_inner[train_m])
        X_te = scaler.transform(X_inner[test_m])
        cols = bundle.feature_names

        model = lgb.LGBMRegressor(
            objective="regression",
            random_state=tc.random_seed,
            n_estimators=tc.lgb_n_estimators,
            learning_rate=tc.lgb_learning_rate,
            num_leaves=tc.lgb_num_leaves,
            subsample=tc.lgb_subsample,
            colsample_bytree=tc.lgb_colsample_bytree,
            verbose=tc.lgb_verbose,
        )
        fit_callbacks = [lgb.early_stopping(tc.lgb_early_stopping_rounds, verbose=False)]
        fit_callbacks.extend(eval_cb)
        model.fit(
            _as_lgb_frame(X_tr, cols),
            y_inner[train_m],
            eval_set=[(_as_lgb_frame(X_te, cols), y_inner[test_m])],
            eval_metric="mae",
            callbacks=fit_callbacks,
        )
        pred = model.predict(_as_lgb_frame(X_te, cols))
        oof_pred[np.where(test_m)[0]] = pred

        mae = float(np.mean(np.abs(pred - y_inner[test_m])))
        da = directional_accuracy(y_inner[test_m], pred)
        bt = aggregate_fold_backtest(
            DatasetBundle(
                X=X_inner[test_m],
                y=y_inner[test_m],
                window_id=wid_inner[test_m],
                t_index=t_inner[test_m],
                global_idx=g_inner[test_m],
                ts_close_ms=ts_inner[test_m],
                feature_names=bundle.feature_names,
                close=bundle.close,
                open_time_ms=bundle.open_time_ms,
            ),
            pred,
            tc.backtest_decision_t,
        )
        sharpe = _annualized_sharpe(bt["pnl_frac"].to_numpy(), tc.periods_per_year)

        fold_rows.append(
            {
                "fold": fold_id,
                "n_train_windows": int(np.unique(tr_w).size),
                "n_test_windows": int(np.unique(te_w).size),
                "mae": mae,
                "directional_accuracy": da,
                "sharpe_annualized": float(sharpe),
            }
        )
        bi = getattr(model, "best_iteration_", None)
        bi_s = str(bi) if bi is not None else "n/a"
        _log(
            f"  Fold {fold_id + 1} done in {time.perf_counter() - t_fold:.1f}s — "
            f"MAE={mae:.6f}  dir_acc={da:.4f}  sharpe~={sharpe:.2f}  (best_iter={bi_s})"
        )

    valid = np.isfinite(oof_pred)
    overall_mae = float(np.mean(np.abs(oof_pred[valid] - y_inner[valid])))
    overall_da = directional_accuracy(y_inner[valid], oof_pred[valid])
    bt_all = aggregate_fold_backtest(
        DatasetBundle(
            X=X_inner[valid],
            y=y_inner[valid],
            window_id=wid_inner[valid],
            t_index=t_inner[valid],
            global_idx=g_inner[valid],
            ts_close_ms=ts_inner[valid],
            feature_names=bundle.feature_names,
            close=bundle.close,
            open_time_ms=bundle.open_time_ms,
        ),
        oof_pred[valid],
        tc.backtest_decision_t,
    )
    overall_sharpe = _annualized_sharpe(bt_all["pnl_frac"].to_numpy(), tc.periods_per_year)

    metrics: Dict[str, Any] = {
        "walk_forward_folds": fold_rows,
        "oof_mae": overall_mae,
        "oof_directional_accuracy": overall_da,
        "oof_sharpe_annualized": float(overall_sharpe),
        "model_type": "lightgbm",
    }
    return metrics, oof_pred


def run_walk_forward_lstm(
    bundle: DatasetBundle,
    cfg: Config,
    inner_unique_ids: np.ndarray,
    device: str,
    *,
    verbose_epochs: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray]:
    assert torch is not None
    tc = cfg.train
    X = bundle.X
    y = bundle.y
    wid = bundle.window_id
    gidx = bundle.global_idx

    inner_mask = np.isin(wid, inner_unique_ids)
    X_inner = X[inner_mask]
    y_inner = y[inner_mask]
    wid_inner = wid[inner_mask]
    gidx_inner = gidx[inner_mask]
    t_inner = bundle.t_index[inner_mask]

    X_seq = build_return_sequences(bundle.close, gidx_inner, tc.lstm_seq_len)

    unique_sorted = np.sort(np.unique(wid_inner))
    folds = walk_forward_masks(unique_sorted, tc.walk_forward_splits, tc.min_train_windows)
    _log(f"LSTM walk-forward: {len(folds)} fold(s), seq_len={tc.lstm_seq_len}, inner rows={X_inner.shape[0]:,}")

    oof_pred = np.full(X_inner.shape[0], np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, Any]] = []

    for fold_id, (tr_w, te_w) in enumerate(folds):
        train_m = np.isin(wid_inner, tr_w)
        test_m = np.isin(wid_inner, te_w)
        _log(
            f"LSTM fold {fold_id + 1}/{len(folds)}: train rows={int(train_m.sum()):,}, "
            f"test rows={int(test_m.sum()):,} …"
        )
        t_fold = time.perf_counter()
        model, scaler = train_lstm(
            X_seq, y_inner, train_m, test_m, tc, device, verbose=verbose_epochs
        )
        pred = predict_lstm(model, scaler, X_seq[test_m], device)
        oof_pred[np.where(test_m)[0]] = pred

        mae = float(np.mean(np.abs(pred - y_inner[test_m])))
        da = directional_accuracy(y_inner[test_m], pred)
        bt = aggregate_fold_backtest(
            DatasetBundle(
                X=X_inner[test_m],
                y=y_inner[test_m],
                window_id=wid_inner[test_m],
                t_index=t_inner[test_m],
                global_idx=gidx_inner[test_m],
                ts_close_ms=bundle.ts_close_ms[inner_mask][test_m],
                feature_names=bundle.feature_names,
                close=bundle.close,
                open_time_ms=bundle.open_time_ms,
            ),
            pred,
            tc.backtest_decision_t,
        )
        sharpe = _annualized_sharpe(bt["pnl_frac"].to_numpy(), tc.periods_per_year)
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train_windows": int(np.unique(tr_w).size),
                "n_test_windows": int(np.unique(te_w).size),
                "mae": mae,
                "directional_accuracy": da,
                "sharpe_annualized": float(sharpe),
            }
        )
        _log(
            f"  LSTM fold {fold_id + 1} done in {time.perf_counter() - t_fold:.1f}s — "
            f"MAE={mae:.6f}  dir_acc={da:.4f}  sharpe~={sharpe:.2f}"
        )

    valid = np.isfinite(oof_pred)
    overall_mae = float(np.mean(np.abs(oof_pred[valid] - y_inner[valid])))
    overall_da = directional_accuracy(y_inner[valid], oof_pred[valid])
    bt_all = aggregate_fold_backtest(
        DatasetBundle(
            X=X_inner[valid],
            y=y_inner[valid],
            window_id=wid_inner[valid],
            t_index=t_inner[valid],
            global_idx=gidx_inner[valid],
            ts_close_ms=bundle.ts_close_ms[inner_mask][valid],
            feature_names=bundle.feature_names,
            close=bundle.close,
            open_time_ms=bundle.open_time_ms,
        ),
        oof_pred[valid],
        tc.backtest_decision_t,
    )
    overall_sharpe = _annualized_sharpe(bt_all["pnl_frac"].to_numpy(), tc.periods_per_year)

    metrics: Dict[str, Any] = {
        "walk_forward_folds": fold_rows,
        "oof_mae": overall_mae,
        "oof_directional_accuracy": overall_da,
        "oof_sharpe_annualized": float(overall_sharpe),
        "model_type": "lstm",
    }
    return metrics, oof_pred


def _as_lgb_frame(X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=feature_names)


def fit_final_lightgbm(
    bundle: DatasetBundle,
    train_mask: np.ndarray,
    cfg: Config,
    *,
    lgb_log_period: int = 50,
) -> Tuple[lgb.LGBMRegressor, StandardScaler]:
    assert lgb is not None
    tc = cfg.train
    n_tr = int(train_mask.sum())
    _log(f"Final LightGBM fit on {n_tr:,} rows (all inner windows, no holdout) …")
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(bundle.X[train_mask])
    y_tr = bundle.y[train_mask]
    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=tc.random_seed,
        n_estimators=tc.lgb_n_estimators,
        learning_rate=tc.lgb_learning_rate,
        num_leaves=tc.lgb_num_leaves,
        subsample=tc.lgb_subsample,
        colsample_bytree=tc.lgb_colsample_bytree,
        verbose=tc.lgb_verbose,
    )
    cb = _lgb_eval_callbacks(lgb_log_period)
    fit_kw: Dict[str, Any] = {}
    if cb:
        fit_kw["callbacks"] = cb
    model.fit(_as_lgb_frame(X_tr, bundle.feature_names), y_tr, **fit_kw)
    _log(f"Final model fit finished in {time.perf_counter() - t0:.1f}s (n_estimators={tc.lgb_n_estimators}).")
    return model, scaler


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    global _QUIET
    parser = argparse.ArgumentParser(description="Train intra-window BTC 5m return model.")
    parser.add_argument("--model", choices=["lightgbm", "lstm"], default=None, help="Override config model_type.")
    parser.add_argument("--device", default="cpu", help="torch device for LSTM (cuda or cpu).")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output (no progress).")
    parser.add_argument(
        "--lgb-log-period",
        type=int,
        default=50,
        metavar="N",
        help="Print LightGBM validation MAE every N boosting iterations (0=off). Default 50.",
    )
    parser.add_argument(
        "--dataset-progress-every",
        type=int,
        default=500,
        metavar="W",
        help="While building X/y, log every W windows (0=no per-window progress). Default 500.",
    )
    args = parser.parse_args(argv)

    _QUIET = bool(args.quiet)
    cfg = Config()
    cfg.ensure_dirs()
    if args.model is not None:
        cfg.train.model_type = args.model

    if cfg.train.model_type == "lightgbm" and lgb is None:
        raise RuntimeError("lightgbm is not installed.")
    if cfg.train.model_type == "lstm" and torch is None:
        raise RuntimeError("torch is not installed.")

    _log(f"Starting training | model={cfg.train.model_type} | data={cfg.paths.data_dir} | glob={cfg.data.csv_glob}")
    t_all = time.perf_counter()
    t_ds = time.perf_counter()
    bundle = build_dataset(
        cfg,
        verbose=not _QUIET,
        progress_every_windows=max(0, int(args.dataset_progress_every)),
    )
    _log(f"Dataset ready in {time.perf_counter() - t_ds:.1f}s.")

    # Window ids are 0..K-1 in time order.
    all_wids = np.unique(bundle.window_id)
    n_w = all_wids.size
    hold_n = max(1, int(n_w * cfg.train.final_holdout_fraction))
    inner_wids = all_wids[:-hold_n]
    hold_wids = all_wids[-hold_n:]

    inner_mask = np.isin(bundle.window_id, inner_wids)
    hold_mask = np.isin(bundle.window_id, hold_wids)

    metrics: Dict[str, Any] = {
        "n_windows_total": int(n_w),
        "n_windows_inner": int(inner_wids.size),
        "n_windows_holdout": int(hold_wids.size),
        "window_size": cfg.data.window_size,
        "lookback_context": cfg.data.lookback_context,
        "feature_names": bundle.feature_names,
    }

    oof_inner = np.full(bundle.X.shape[0], np.nan, dtype=np.float64)

    lgb_period = max(0, int(args.lgb_log_period))
    t_wf = time.perf_counter()
    if cfg.train.model_type == "lightgbm":
        wf_metrics, oof_vec = run_walk_forward_lightgbm(
            bundle, cfg, inner_wids, lgb_log_period=lgb_period
        )
        metrics.update(wf_metrics)
        # map oof_vec (length n_inner_rows) back to global indices
        inner_positions = np.where(inner_mask)[0]
        # oof_vec order matches X_inner row order
        oof_inner[inner_positions] = oof_vec
    else:
        wf_metrics, oof_vec = run_walk_forward_lstm(
            bundle, cfg, inner_wids, args.device, verbose_epochs=not _QUIET
        )
        inner_positions = np.where(inner_mask)[0]
        oof_inner[inner_positions] = oof_vec

    _log(f"Walk-forward finished in {time.perf_counter() - t_wf:.1f}s. Fitting production model …")

    # Final production fit on all inner windows (excluding final holdout)
    train_prod_mask = inner_mask & np.isfinite(bundle.X).all(axis=1)
    if cfg.train.model_type == "lightgbm":
        final_model, scaler = fit_final_lightgbm(
            bundle, train_prod_mask, cfg, lgb_log_period=lgb_period
        )
        model_path = cfg.paths.outputs_dir / "lgb_model.txt"
        scaler_path = cfg.paths.outputs_dir / "scaler.joblib"
        feat_path = cfg.paths.outputs_dir / "feature_names.json"
        final_model.booster_.save_model(str(model_path))
        joblib.dump(scaler, scaler_path)
        save_json(feat_path, bundle.feature_names)

        # Holdout evaluation (fit scaler+model only on inner — already fit; evaluate on holdout)
        X_hold = scaler.transform(bundle.X[hold_mask])
        pred_hold = final_model.predict(_as_lgb_frame(X_hold, bundle.feature_names))
        hold_mae = float(np.mean(np.abs(pred_hold - bundle.y[hold_mask])))
        hold_da = directional_accuracy(bundle.y[hold_mask], pred_hold)
        hold_bt = aggregate_fold_backtest(
            DatasetBundle(
                X=bundle.X[hold_mask],
                y=bundle.y[hold_mask],
                window_id=bundle.window_id[hold_mask],
                t_index=bundle.t_index[hold_mask],
                global_idx=bundle.global_idx[hold_mask],
                ts_close_ms=bundle.ts_close_ms[hold_mask],
                feature_names=bundle.feature_names,
                close=bundle.close,
                open_time_ms=bundle.open_time_ms,
            ),
            pred_hold,
            cfg.train.backtest_decision_t,
        )
        hold_sharpe = _annualized_sharpe(hold_bt["pnl_frac"].to_numpy(), cfg.train.periods_per_year)
        metrics["holdout_mae"] = hold_mae
        metrics["holdout_directional_accuracy"] = hold_da
        metrics["holdout_sharpe_annualized"] = float(hold_sharpe)

        hold_bt_path = cfg.paths.outputs_dir / "backtest_holdout_decision_t.csv"
        hold_bt.to_csv(hold_bt_path, index=False)

        oof_bt = aggregate_fold_backtest(bundle, oof_inner, cfg.train.backtest_decision_t)
        oof_bt_path = cfg.paths.outputs_dir / "backtest_oof_decision_t.csv"
        oof_bt.to_csv(oof_bt_path, index=False)

        metrics_path = cfg.paths.outputs_dir / "metrics.json"
        save_json(metrics_path, metrics)
        _log(f"Artifacts written to {cfg.paths.outputs_dir}  (total time {time.perf_counter() - t_all:.1f}s)")

    else:
        # LSTM: save torch state + seq scaler; reuse walk-forward metrics only; optional final train
        assert torch is not None
        tc = cfg.train
        _log("Building return sequences for final LSTM …")
        t_seq = time.perf_counter()
        X_seq_full = build_return_sequences(bundle.close, bundle.global_idx, tc.lstm_seq_len)
        _log(f"Sequences shape {X_seq_full.shape} in {time.perf_counter() - t_seq:.1f}s")
        train_m = train_prod_mask
        # small internal val tail from train
        tr_w = np.unique(bundle.window_id[train_m])
        cut = int(tr_w.size * 0.9)
        tr_w_sub = tr_w[:cut]
        va_w_sub = tr_w[cut:]
        sub_train = train_m & np.isin(bundle.window_id, tr_w_sub)
        sub_val = train_m & np.isin(bundle.window_id, va_w_sub)
        _log("Final LSTM train (90/10 window split on inner) …")
        model, scaler = train_lstm(
            X_seq_full, bundle.y, sub_train, sub_val, tc, args.device, verbose=not _QUIET
        )
        torch.save(model.state_dict(), cfg.paths.outputs_dir / "lstm_state.pt")
        joblib.dump(scaler, cfg.paths.outputs_dir / "lstm_seq_scaler.joblib")
        save_json(cfg.paths.outputs_dir / "feature_names.json", {"tabular": bundle.feature_names, "lstm": "return_sequence"})

        pred_hold = predict_lstm(model, scaler, X_seq_full[hold_mask], args.device)
        hold_mae = float(np.mean(np.abs(pred_hold - bundle.y[hold_mask])))
        hold_da = directional_accuracy(bundle.y[hold_mask], pred_hold)
        hold_bt = aggregate_fold_backtest(
            DatasetBundle(
                X=bundle.X[hold_mask],
                y=bundle.y[hold_mask],
                window_id=bundle.window_id[hold_mask],
                t_index=bundle.t_index[hold_mask],
                global_idx=bundle.global_idx[hold_mask],
                ts_close_ms=bundle.ts_close_ms[hold_mask],
                feature_names=bundle.feature_names,
                close=bundle.close,
                open_time_ms=bundle.open_time_ms,
            ),
            pred_hold,
            cfg.train.backtest_decision_t,
        )
        hold_sharpe = _annualized_sharpe(hold_bt["pnl_frac"].to_numpy(), cfg.train.periods_per_year)
        metrics["holdout_mae"] = hold_mae
        metrics["holdout_directional_accuracy"] = hold_da
        metrics["holdout_sharpe_annualized"] = float(hold_sharpe)
        hold_bt.to_csv(cfg.paths.outputs_dir / "backtest_holdout_decision_t.csv", index=False)
        oof_bt = aggregate_fold_backtest(bundle, oof_inner, cfg.train.backtest_decision_t)
        oof_bt.to_csv(cfg.paths.outputs_dir / "backtest_oof_decision_t.csv", index=False)
        save_json(cfg.paths.outputs_dir / "metrics.json", metrics)
        _log(f"Artifacts written to {cfg.paths.outputs_dir}  (total time {time.perf_counter() - t_all:.1f}s)")

    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
