"""
Load trained LightGBM regressor + sklearn scaler and run tabular inference.

Example:
  from pathlib import Path
  from inference import load_lightgbm_bundle, predict_final_window_pct

  model, scaler, names = load_lightgbm_bundle(Path("models/artifacts"))
  X = ...  # shape (n_rows, len(names)), same order as training
  Xs = scaler.transform(X)
  pred = predict_final_window_pct(model, Xs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    lgb = None
    _import_error = e
else:
    _import_error = None


def load_lightgbm_bundle(artifacts_dir: Path) -> Tuple["lgb.Booster", object, List[str]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for inference") from _import_error
    import joblib

    artifacts_dir = Path(artifacts_dir)
    booster = lgb.Booster(model_file=str(artifacts_dir / "lgb_model.txt"))
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    names = json.loads((artifacts_dir / "feature_names.json").read_text(encoding="utf-8"))
    if isinstance(names, dict) and "tabular" in names:
        names = list(names["tabular"])
    return booster, scaler, list(names)


def predict_final_window_pct(booster: "lgb.Booster", X_scaled: np.ndarray) -> np.ndarray:
    """Return predicted full-window close-to-open percent move."""
    return booster.predict(X_scaled)
