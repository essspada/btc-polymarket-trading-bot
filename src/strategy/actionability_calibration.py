from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clamp_prob(value: float) -> float:
    return float(max(1e-6, min(1.0 - 1e-6, value)))


def wilson_lower_bound(*, wins: int, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    phat = float(wins) / float(n)
    z2 = float(z) ** 2
    denom = 1.0 + (z2 / float(n))
    center = phat + (z2 / (2.0 * float(n)))
    margin = float(z) * math.sqrt((phat * (1.0 - phat) + (z2 / (4.0 * float(n)))) / float(n))
    return float(max(0.0, min(1.0, (center - margin) / denom)))


@dataclass(frozen=True)
class CalibrationResult:
    applied: bool
    rejected: bool
    reason: str
    raw_p_side: float
    calibrated_p_side: float
    calibrated_net_edge: float | None
    calibrated_breakeven_margin: float | None
    calibrated_expected_roi_cash: float | None
    lower_bound_sources: list[dict[str, Any]]
    lower_bound_floor: float | None


@lru_cache(maxsize=64)
def load_actionability_artifact(path: str) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _bucket_match(
    *,
    feature: str,
    value: float,
    buckets: list[dict[str, Any]],
    min_bucket_samples: int,
) -> tuple[float | None, dict[str, Any] | None]:
    for bucket in buckets:
        lower = _float(bucket.get("lower"))
        upper = _float(bucket.get("upper"))
        if lower is None or upper is None:
            continue
        if lower <= value < upper or (value == upper and math.isclose(upper, 1.0)):
            n = int(_float(bucket.get("n")) or 0)
            if n < int(min_bucket_samples):
                return None, {
                    "feature": feature,
                    "value": float(value),
                    "lower": float(lower),
                    "upper": float(upper),
                    "n": n,
                    "status": "insufficient_samples",
                }
            lb = _float(bucket.get("lb_wr"))
            if lb is None:
                return None, {
                    "feature": feature,
                    "value": float(value),
                    "lower": float(lower),
                    "upper": float(upper),
                    "n": n,
                    "status": "missing_lb_wr",
                }
            return float(lb), {
                "feature": feature,
                "value": float(value),
                "lower": float(lower),
                "upper": float(upper),
                "n": n,
                "lb_wr": float(lb),
                "status": "ok",
            }
    return None, None


def evaluate_actionability_calibration(
    *,
    raw_p_side: float,
    cash_per_share: float | None,
    breakeven_probability: float | None,
    size_shares: float | None,
    fill_probability: float | None,
    cash_required: float | None,
    context: Mapping[str, Any] | None,
    calibration_cfg: Mapping[str, Any] | None,
) -> CalibrationResult | None:
    cfg = calibration_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return None

    artifact: dict[str, Any] | None = None
    if isinstance(cfg.get("artifact"), Mapping):
        artifact = dict(cfg.get("artifact") or {})
    elif cfg.get("artifact_path"):
        artifact = load_actionability_artifact(str(cfg.get("artifact_path")))
    if not artifact:
        return CalibrationResult(
            applied=False,
            rejected=True,
            reason="actionability_artifact_missing",
            raw_p_side=float(raw_p_side),
            calibrated_p_side=float(raw_p_side),
            calibrated_net_edge=None,
            calibrated_breakeven_margin=None,
            calibrated_expected_roi_cash=None,
            lower_bound_sources=[],
            lower_bound_floor=None,
        )

    fallback_policy = str(cfg.get("fallback_policy") or "conservative_shrink").strip().lower()
    min_bucket_samples = int(_float(cfg.get("min_bucket_samples")) or _float(artifact.get("min_bucket_samples")) or 20)
    conservative_alpha = float(_float(cfg.get("conservative_alpha")) or _float(artifact.get("conservative_alpha")) or 0.7)
    conservative_alpha = max(0.0, min(1.0, conservative_alpha))

    global_lb = _float(((artifact.get("global") or {}) if isinstance(artifact.get("global"), Mapping) else {}).get("lb_wr"))
    lower_bounds: list[float] = []
    sources: list[dict[str, Any]] = []
    missing_features = 0
    insufficient_features = 0

    features = artifact.get("features")
    if isinstance(features, Mapping):
        for feature, payload in features.items():
            if not isinstance(payload, Mapping):
                continue
            value = _float((context or {}).get(str(feature)))
            if value is None:
                missing_features += 1
                continue
            buckets = payload.get("buckets")
            if not isinstance(buckets, list):
                continue
            lb, source = _bucket_match(
                feature=str(feature),
                value=float(value),
                buckets=[dict(item) for item in buckets if isinstance(item, Mapping)],
                min_bucket_samples=min_bucket_samples,
            )
            if source is not None:
                sources.append(source)
            if lb is None:
                if source is not None and source.get("status") == "insufficient_samples":
                    insufficient_features += 1
                continue
            lower_bounds.append(float(lb))

    raw = float(_clamp_prob(raw_p_side))
    fallback_floor = _clamp_prob(0.5 + conservative_alpha * (raw - 0.5))
    if global_lb is not None:
        fallback_floor = min(raw, max(fallback_floor, _clamp_prob(float(global_lb))))

    if lower_bounds:
        lb_floor = min(lower_bounds)
        calibrated_p_side = min(raw, _clamp_prob(lb_floor))
        reason = "calibrated"
    else:
        lb_floor = None
        if fallback_policy == "reject":
            reason = "actionability_insufficient_bucket_support"
            calibrated_p_side = raw
        else:
            reason = "calibrated_fallback_conservative_shrink"
            calibrated_p_side = min(raw, fallback_floor)

    cp_share = _float(cash_per_share)
    be_prob = _float(breakeven_probability)
    size = _float(size_shares)
    fill = _float(fill_probability)
    cash = _float(cash_required)
    calibrated_net_edge: float | None = None
    calibrated_breakeven_margin: float | None = None
    calibrated_expected_roi_cash: float | None = None
    if cp_share is not None:
        calibrated_net_edge = float(calibrated_p_side - cp_share)
    if be_prob is not None:
        calibrated_breakeven_margin = float(calibrated_p_side - be_prob)
    if size is not None and size > 0.0 and fill is not None and cash is not None and cash > 0.0 and cp_share is not None:
        calibrated_ev_fill = (calibrated_p_side - cp_share) * size
        calibrated_ev_exec = calibrated_ev_fill * fill
        calibrated_expected_roi_cash = float(calibrated_ev_exec / cash)

    min_calibrated_net_edge = _float(cfg.get("min_calibrated_net_edge"))
    min_breakeven_margin = _float(cfg.get("min_breakeven_margin"))

    reject_reason: str | None = None
    if fallback_policy == "reject" and not lower_bounds:
        reject_reason = "actionability_insufficient_bucket_support"
    elif min_calibrated_net_edge is not None and (
        calibrated_net_edge is None or float(calibrated_net_edge) < float(min_calibrated_net_edge)
    ):
        reject_reason = "calibrated_net_edge_below_threshold"
    if reject_reason is None and min_breakeven_margin is not None and (
        calibrated_breakeven_margin is None or float(calibrated_breakeven_margin) < float(min_breakeven_margin)
    ):
        reject_reason = "calibrated_breakeven_margin_below_threshold"

    if reject_reason is not None:
        return CalibrationResult(
            applied=True,
            rejected=True,
            reason=reject_reason,
            raw_p_side=raw,
            calibrated_p_side=float(calibrated_p_side),
            calibrated_net_edge=calibrated_net_edge,
            calibrated_breakeven_margin=calibrated_breakeven_margin,
            calibrated_expected_roi_cash=calibrated_expected_roi_cash,
            lower_bound_sources=sources,
            lower_bound_floor=lb_floor,
        )

    # Surface weak-support telemetry even when fallback shrink is allowed.
    if not lower_bounds and (missing_features > 0 or insufficient_features > 0):
        reason = f"{reason}:missing={missing_features},insufficient={insufficient_features}"

    return CalibrationResult(
        applied=True,
        rejected=False,
        reason=reason,
        raw_p_side=raw,
        calibrated_p_side=float(calibrated_p_side),
        calibrated_net_edge=calibrated_net_edge,
        calibrated_breakeven_margin=calibrated_breakeven_margin,
        calibrated_expected_roi_cash=calibrated_expected_roi_cash,
        lower_bound_sources=sources,
        lower_bound_floor=lb_floor,
    )
