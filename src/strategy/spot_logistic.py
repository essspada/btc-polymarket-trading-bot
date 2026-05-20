from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.strategy.calibration import clip_prob


@dataclass
class SpotLogisticConfig:
    lookback: int = 240
    min_train_samples: int = 40
    min_class_samples: int = 10

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpotLogisticConfig:
        payload = raw or {}
        return cls(
            lookback=max(20, int(payload.get("lookback", 240))),
            min_train_samples=max(10, int(payload.get("min_train_samples", 40))),
            min_class_samples=max(2, int(payload.get("min_class_samples", 10))),
        )


def _label(row: dict[str, Any]) -> int | None:
    side = str(row.get("actual_side", "")).strip().lower()
    if side == "up":
        return 1
    if side == "down":
        return 0
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("market_id") or ""),
        str(row.get("created_at") or row.get("ts_utc") or row.get("resolved_ts_utc") or ""),
    )


def row_timestamp(row: dict[str, Any]) -> datetime | None:
    for key in ("created_at", "ts_utc", "resolved_ts_utc", "end_time"):
        ts = _parse_ts(row.get(key))
        if ts is not None:
            return ts
    return None


def feature_row(row: dict[str, Any]) -> list[float]:
    market_p = row.get("market_p_up")
    has_market = 1.0 if market_p is not None else 0.0
    market_prob = clip_prob(_to_float(market_p, 0.5))
    proxy_prob = clip_prob(_to_float(row.get("proxy_p_up"), 0.5))
    return [
        _to_float(row.get("spot_return_bps_from_open"), 0.0) / 100.0,
        _to_float(row.get("spot_recent_return_1m_bps"), 0.0) / 100.0,
        _to_float(row.get("spot_recent_vol_5m_bps"), 0.0) / 100.0,
        _to_float(row.get("seconds_to_expiry"), 300.0) / 300.0,
        market_prob,
        has_market,
        proxy_prob,
    ]


def fit_rolling_spot_logistic(rows: Iterable[dict[str, Any]], cfg: SpotLogisticConfig) -> Pipeline | None:
    data = list(rows)
    if cfg.lookback > 0:
        data = data[-cfg.lookback :]

    x: list[list[float]] = []
    y: list[int] = []
    for row in data:
        label = _label(row)
        if label is None:
            continue
        if row.get("spot_return_bps_from_open") is None:
            continue
        x.append(feature_row(row))
        y.append(label)

    if len(y) < cfg.min_train_samples:
        return None
    ones = sum(y)
    zeros = len(y) - ones
    if min(ones, zeros) < cfg.min_class_samples:
        return None

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
    return model


def predict_spot_logistic_probability(model: Pipeline | None, row: dict[str, Any]) -> float | None:
    if model is None:
        return None
    if row.get("spot_return_bps_from_open") is None:
        return None
    x = np.asarray([feature_row(row)], dtype=float)
    prob = float(model.predict_proba(x)[0, 1])
    return clip_prob(prob)


def build_causal_spot_logistic_probability_map(
    rows: Iterable[dict[str, Any]],
    cfg: SpotLogisticConfig,
    *,
    seed_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[tuple[str, str], float]:
    target_rows = list(rows)
    seed_rows_list = list(seed_rows or [])
    target_rows.sort(key=lambda row: (row_timestamp(row) or datetime.min.replace(tzinfo=UTC), row_key(row)))
    seed_rows_list.sort(key=lambda row: (row_timestamp(row) or datetime.min.replace(tzinfo=UTC), row_key(row)))

    out: dict[tuple[str, str], float] = {}
    active_seed: list[dict[str, Any]] = []
    active_seed_keys: set[tuple[str, str]] = set()
    observed_rows: list[dict[str, Any]] = []
    observed_row_keys: set[tuple[str, str]] = set()
    seed_idx = 0

    for row in target_rows:
        current_ts = row_timestamp(row)
        key = row_key(row)

        while seed_idx < len(seed_rows_list):
            seed_row = seed_rows_list[seed_idx]
            seed_ts = row_timestamp(seed_row)
            if current_ts is None or seed_ts is None or seed_ts >= current_ts:
                break
            seed_key = row_key(seed_row)
            if seed_key not in active_seed_keys:
                active_seed.append(seed_row)
                active_seed_keys.add(seed_key)
            seed_idx += 1

        train_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for train_row in active_seed + observed_rows:
            train_key = row_key(train_row)
            if train_key in seen_keys:
                continue
            train_rows.append(train_row)
            seen_keys.add(train_key)

        model = fit_rolling_spot_logistic(train_rows, cfg) if train_rows else None
        prob = predict_spot_logistic_probability(model, row)
        if prob is not None:
            out[key] = prob

        if _label(row) is not None and row.get("spot_return_bps_from_open") is not None and key not in observed_row_keys:
            observed_rows.append(row)
            observed_row_keys.add(key)

    return out
