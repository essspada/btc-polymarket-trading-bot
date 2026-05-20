from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        lr = LogisticRegression(random_state=42, max_iter=500)
        lr.fit(x, y)
        return cls(model=lr)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


class IdentityCalibrator:
    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float)


def clip_prob(p: float) -> float:
    return float(np.clip(float(p), 0.01, 0.99))


def _actual_label(row: dict[str, Any]) -> int | None:
    side = str(row.get("actual_side", "")).strip().lower()
    if side == "up":
        return 1
    if side == "down":
        return 0
    return None


def fit_rolling_platt(
    rows: Iterable[dict[str, Any]],
    prob_key: str = "p_up",
    lookback: int = 240,
    min_samples: int = 40,
    min_class_samples: int = 8,
) -> PlattCalibrator | None:
    x_vals: list[float] = []
    y_vals: list[int] = []
    for row in rows:
        y = _actual_label(row)
        if y is None:
            continue
        try:
            p = clip_prob(float(row.get(prob_key)))
        except Exception:
            continue
        x_vals.append(p)
        y_vals.append(y)

    if lookback > 0 and len(x_vals) > lookback:
        x_vals = x_vals[-lookback:]
        y_vals = y_vals[-lookback:]
    if len(x_vals) < max(1, int(min_samples)):
        return None

    y_arr = np.asarray(y_vals, dtype=int)
    pos = int(y_arr.sum())
    neg = int(len(y_arr) - pos)
    if pos < int(min_class_samples) or neg < int(min_class_samples):
        return None

    return PlattCalibrator.fit(np.asarray(x_vals, dtype=float), y_arr)


def calibrate_probability(p: float, calibrator: PlattCalibrator | IdentityCalibrator | None) -> float:
    if calibrator is None:
        return clip_prob(p)
    pred = calibrator.predict(np.asarray([clip_prob(p)], dtype=float))
    if pred.size <= 0:
        return clip_prob(p)
    return clip_prob(float(pred[0]))


def compute_online_bias_shift(
    rows: Iterable[dict[str, Any]],
    prob_key: str = "p_up",
    lookback: int = 120,
    min_samples: int = 20,
    strength: float = 0.7,
    max_abs_shift: float = 0.15,
) -> float:
    data: list[tuple[float, int]] = []
    for row in rows:
        y = _actual_label(row)
        if y is None:
            continue
        try:
            p = clip_prob(float(row.get(prob_key)))
        except Exception:
            continue
        data.append((p, y))

    if lookback > 0 and len(data) > lookback:
        data = data[-lookback:]
    n = len(data)
    if n < max(1, int(min_samples)):
        return 0.0

    mean_p = float(np.mean([x[0] for x in data]))
    mean_y = float(np.mean([x[1] for x in data]))
    raw_shift = mean_y - mean_p
    shrink = n / (n + 25.0)
    shift = float(strength) * raw_shift * shrink
    lim = abs(float(max_abs_shift))
    return float(np.clip(shift, -lim, lim))


def compute_adaptive_model_weight(
    rows: Iterable[dict[str, Any]],
    model_prob_key: str = "model_p_up",
    market_prob_key: str = "market_p_up",
    lookback: int = 120,
    min_samples: int = 20,
    default_weight: float = 0.35,
    min_weight: float = 0.1,
    max_weight: float = 0.9,
) -> float:
    data: list[tuple[float, float, int]] = []
    for row in rows:
        y = _actual_label(row)
        if y is None:
            continue
        try:
            p_model = clip_prob(float(row.get(model_prob_key)))
            p_market = clip_prob(float(row.get(market_prob_key)))
        except Exception:
            continue
        data.append((p_model, p_market, y))

    if lookback > 0 and len(data) > lookback:
        data = data[-lookback:]
    n = len(data)
    base = float(np.clip(float(default_weight), min_weight, max_weight))
    if n < max(1, int(min_samples)):
        return base

    b_model = float(np.mean([(m - y) ** 2 for m, _, y in data]))
    b_market = float(np.mean([(k - y) ** 2 for _, k, y in data]))

    inv_model = 1.0 / max(1e-9, b_model)
    inv_market = 1.0 / max(1e-9, b_market)
    perf_weight = inv_model / (inv_model + inv_market)

    shrink = n / (n + 30.0)
    w = (1.0 - shrink) * base + shrink * perf_weight
    return float(np.clip(w, min_weight, max_weight))
