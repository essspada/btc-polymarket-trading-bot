from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.strategy.calibration import clip_prob


@dataclass
class SpotConsensusConfig:
    path_weight: float = 0.5
    logistic_weight: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SpotConsensusConfig:
        payload = raw or {}
        return cls(
            path_weight=max(0.0, float(payload.get("path_weight", 0.5))),
            logistic_weight=max(0.0, float(payload.get("logistic_weight", 0.5))),
        )


def predict_spot_consensus_blend_probability(
    *,
    spot_window_path_p_up: float | None,
    spot_logistic_p_up: float | None,
    spot_market_blend_p_up: float | None,
    market_p_up: float | None = None,
    cfg: SpotConsensusConfig | None = None,
) -> float | None:
    config = cfg or SpotConsensusConfig()
    path_p = float(spot_window_path_p_up) if spot_window_path_p_up is not None else None
    logistic_p = float(spot_logistic_p_up) if spot_logistic_p_up is not None else None
    blend_p = float(spot_market_blend_p_up) if spot_market_blend_p_up is not None else None
    market_p = float(market_p_up) if market_p_up is not None else None

    if path_p is not None and logistic_p is not None:
        path_side = path_p >= 0.5
        logistic_side = logistic_p >= 0.5
        if path_side == logistic_side:
            total = float(config.path_weight + config.logistic_weight)
            if total <= 0.0:
                return clip_prob((path_p + logistic_p) / 2.0)
            return clip_prob((config.path_weight * path_p + config.logistic_weight * logistic_p) / total)

    for value in (blend_p, logistic_p, path_p, market_p):
        if value is not None:
            return clip_prob(value)
    return None
