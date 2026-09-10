"""Per-cell demand forecast: a gradient-boosted model, and the baseline it must beat.

THE BASELINE MATTERS MORE THAN THE MODEL. Demand in this data is overwhelmingly periodic -
the same zone at the same hour on the same weekday looks much like it did last week. So the
honest baseline is the historical mean for that (zone, dow, hour) cell, and a
gradient-boosted model has to beat THAT, not beat a global average.

This is the discipline the spec's "the model is the payload, never the point" is asking
for: the model is allowed in only if it earns its place against the arithmetic.

Features are deliberately cyclical-encoded. Hour 23 and hour 0 are adjacent, and a model
given raw integers has to spend splits learning that 23 is next to 0. sin/cos encoding
makes it structural instead.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ForecastResult:
    model: str
    mae: float
    rmse: float
    mape: Optional[float]
    r2: float
    n: int
    beats_baseline: Optional[bool] = None
    improvement_over_baseline: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {"model": self.model, "mae": round(self.mae, 4),
                "rmse": round(self.rmse, 4),
                "mape": round(self.mape, 4) if self.mape is not None else None,
                "r2": round(self.r2, 4), "n": self.n,
                "beats_baseline": self.beats_baseline,
                "improvement_over_baseline_pct": (
                    round(self.improvement_over_baseline * 100, 2)
                    if self.improvement_over_baseline is not None else None)}


def metrics(actual: np.ndarray, predicted: np.ndarray, name: str) -> ForecastResult:
    error = predicted - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # MAPE only over non-zero actuals. Many cells are genuinely zero (nobody hails a cab
    # in that zone at 4am) and dividing by them produces infinities that swamp the metric.
    nonzero = actual > 0
    mape = (float(np.mean(np.abs(error[nonzero] / actual[nonzero]))) * 100
            if nonzero.any() else None)

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return ForecastResult(name, mae, rmse, mape, r2, len(actual))


def cyclical(values: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """sin/cos encoding, so hour 23 sits next to hour 0 by construction."""
    angle = 2 * math.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_features(rows: Sequence[Dict[str, object]]) -> Tuple[np.ndarray, List[str]]:
    hour = np.array([r["hour"] for r in rows], dtype=float)
    dow = np.array([r["dow"] for r in rows], dtype=float)

    hour_sin, hour_cos = cyclical(hour, 24)
    dow_sin, dow_cos = cyclical(dow, 7)

    columns = [
        hour_sin, hour_cos, dow_sin, dow_cos,
        (dow >= 1) & (dow <= 5),                                     # weekday
        np.array([r.get("is_airport", 0) for r in rows], dtype=float),
        np.array([r.get("zone_mean_trips", 0.0) for r in rows], dtype=float),
        np.array([r.get("zone_mean_revenue_per_hour", 0.0) for r in rows], dtype=float),
        np.array([r.get("cell_history_mean", 0.0) for r in rows], dtype=float),
        np.array([r.get("cell_history_sd", 0.0) for r in rows], dtype=float),
    ]
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekday", "is_airport",
             "zone_mean_trips", "zone_mean_revenue_per_hour",
             "cell_history_mean", "cell_history_sd"]
    return np.column_stack([np.asarray(c, dtype=float) for c in columns]), names


def historical_mean_baseline(train: Sequence[Dict[str, object]],
                             test: Sequence[Dict[str, object]]) -> np.ndarray:
    """Predict each cell by its own training mean. Falls back to the zone mean, then the
    global mean, for cells never seen in training - which is what makes it a fair
    baseline rather than one that gets to skip the hard rows."""
    cell: Dict[Tuple[int, int, int], List[float]] = {}
    zone: Dict[int, List[float]] = {}
    for row in train:
        key = (int(row["zone_id"]), int(row["dow"]), int(row["hour"]))
        cell.setdefault(key, []).append(float(row["target"]))
        zone.setdefault(int(row["zone_id"]), []).append(float(row["target"]))

    cell_mean = {k: statistics.fmean(v) for k, v in cell.items()}
    zone_mean = {k: statistics.fmean(v) for k, v in zone.items()}
    overall = statistics.fmean([float(r["target"]) for r in train]) if train else 0.0

    return np.array([
        cell_mean.get((int(r["zone_id"]), int(r["dow"]), int(r["hour"])),
                      zone_mean.get(int(r["zone_id"]), overall))
        for r in test], dtype=float)


def fit_gbm(train: Sequence[Dict[str, object]], test: Sequence[Dict[str, object]],
            seed: int = 20260910) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """LightGBM if available, otherwise a scikit-learn gradient booster.

    Both are real gradient-boosted trees; the fallback exists so the pipeline runs on a
    machine without LightGBM rather than skipping the model entirely and quietly
    reporting only the baseline.
    """
    X_train, names = build_features(train)
    X_test, _ = build_features(test)
    y_train = np.array([float(r["target"]) for r in train])

    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=2, verbose=-1)
        model.fit(X_train, y_train)
        importance = dict(zip(names, (model.feature_importances_ /
                                      max(1, model.feature_importances_.sum())).round(4)))
        return model.predict(X_test), {k: float(v) for k, v in importance.items()}
    except (ImportError, OSError):
        # OSError as well as ImportError: LightGBM is a Python wrapper around a compiled
        # library, so on a machine missing libgomp it is installed and importable right
        # up until it dlopen()s and raises OSError. Catching only ImportError turned a
        # missing system package into a hard failure of the entire pipeline rather than
        # the graceful degradation this fallback exists to provide.
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=seed)
        model.fit(X_train, y_train)
        return model.predict(X_test), None


def evaluate(train: Sequence[Dict[str, object]],
             test: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Baseline first, model second, and the model only counts if it beats the baseline."""
    actual = np.array([float(r["target"]) for r in test])

    baseline_pred = historical_mean_baseline(train, test)
    baseline = metrics(actual, baseline_pred, "historical_cell_mean")

    model_pred, importance = fit_gbm(train, test)
    # Demand cannot be negative; a tree ensemble can extrapolate below zero on sparse
    # cells and a negative forecast would corrupt the capacity calculation downstream.
    model_pred = np.clip(model_pred, 0, None)
    model = metrics(actual, model_pred, "gradient_boosted")

    model.beats_baseline = model.mae < baseline.mae
    model.improvement_over_baseline = ((baseline.mae - model.mae) / baseline.mae
                                       if baseline.mae > 0 else 0.0)

    return {
        "baseline": baseline.as_dict(),
        "model": model.as_dict(),
        "verdict": ("the model earns its place" if model.beats_baseline
                    else "the model does NOT beat the historical-mean baseline; "
                         "use the baseline"),
        "feature_importance": importance,
        "train_rows": len(train), "test_rows": len(test),
    }
