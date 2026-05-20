from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

from src.strategy.calibration import clip_prob


@dataclass
class SpotWindowPathConfig:
    min_sigma_bps: float = 4.0
    drift_weight: float = 0.25
    prob_shrink: float = 0.85
    spot_model_weight: float = 0.45

    @classmethod
    def from_dict(cls, raw: dict | None) -> SpotWindowPathConfig:
        payload = raw or {}
        return cls(
            min_sigma_bps=float(payload.get("min_sigma_bps", 4.0)),
            drift_weight=float(payload.get("drift_weight", 0.25)),
            prob_shrink=float(payload.get("prob_shrink", 0.85)),
            spot_model_weight=float(payload.get("spot_model_weight", 0.45)),
        )


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def predict_spot_window_path_probability(
    *,
    spot_return_bps_from_open: float | None,
    spot_recent_return_1m_bps: float | None,
    spot_recent_vol_5m_bps: float | None,
    seconds_to_expiry: float,
    cfg: SpotWindowPathConfig,
) -> float | None:
    if spot_return_bps_from_open is None:
        return None

    remaining_minutes = max(float(seconds_to_expiry) / 60.0, 0.05)
    sigma_1m_bps = max(abs(float(spot_recent_vol_5m_bps or 0.0)), float(cfg.min_sigma_bps))
    sigma_remaining_bps = max(1e-6, sigma_1m_bps * sqrt(remaining_minutes))

    drift_bps = float(cfg.drift_weight) * float(spot_recent_return_1m_bps or 0.0) * remaining_minutes
    projected_return_bps = float(spot_return_bps_from_open) + drift_bps
    z = projected_return_bps / sigma_remaining_bps

    p = _norm_cdf(z)
    shrunk = 0.5 + float(cfg.prob_shrink) * (p - 0.5)
    return clip_prob(shrunk)


def blend_spot_with_market_probability(*, market_p_up: float, spot_p_up: float, cfg: SpotWindowPathConfig) -> float:
    spot_weight = min(max(float(cfg.spot_model_weight), 0.0), 1.0)
    market_weight = 1.0 - spot_weight
    return clip_prob(market_weight * float(market_p_up) + spot_weight * float(spot_p_up))
