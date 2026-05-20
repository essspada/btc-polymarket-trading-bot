from __future__ import annotations

from src.strategy.actionability_calibration import evaluate_actionability_calibration


def _artifact() -> dict:
    return {
        "schema": "actionability_calibration_v1",
        "min_bucket_samples": 5,
        "conservative_alpha": 0.7,
        "global": {"n": 20, "wins": 12, "lb_wr": 0.50},
        "features": {
            "confidence": {
                "buckets": [
                    {"lower": 0.50, "upper": 0.60, "n": 10, "wins": 5, "lb_wr": 0.40},
                    {"lower": 0.60, "upper": 0.80, "n": 15, "wins": 12, "lb_wr": 0.70},
                ]
            }
        },
    }


def test_actionability_calibration_rejects_when_policy_is_reject_and_bucket_support_missing() -> None:
    result = evaluate_actionability_calibration(
        raw_p_side=0.72,
        cash_per_share=0.62,
        breakeven_probability=0.62,
        size_shares=100.0,
        fill_probability=1.0,
        cash_required=62.0,
        context={"confidence": 0.95},
        calibration_cfg={
            "enabled": True,
            "artifact": _artifact(),
            "fallback_policy": "reject",
            "min_bucket_samples": 5,
            "min_calibrated_net_edge": 0.0,
            "min_breakeven_margin": 0.0,
        },
    )

    assert result is not None
    assert result.rejected is True
    assert result.reason == "actionability_insufficient_bucket_support"


def test_actionability_calibration_conservative_fallback_does_not_require_bucket() -> None:
    result = evaluate_actionability_calibration(
        raw_p_side=0.72,
        cash_per_share=0.62,
        breakeven_probability=0.62,
        size_shares=100.0,
        fill_probability=1.0,
        cash_required=62.0,
        context={"confidence": 0.95},
        calibration_cfg={
            "enabled": True,
            "artifact": _artifact(),
            "fallback_policy": "conservative_shrink",
            "conservative_alpha": 0.5,
            "min_bucket_samples": 5,
            "min_calibrated_net_edge": None,
            "min_breakeven_margin": None,
        },
    )

    assert result is not None
    assert result.rejected is False
    assert result.calibrated_p_side < 0.72
    assert result.calibrated_net_edge is not None


def test_actionability_calibration_uses_bucket_lower_bound_and_threshold() -> None:
    result = evaluate_actionability_calibration(
        raw_p_side=0.72,
        cash_per_share=0.62,
        breakeven_probability=0.62,
        size_shares=100.0,
        fill_probability=1.0,
        cash_required=62.0,
        context={"confidence": 0.65},
        calibration_cfg={
            "enabled": True,
            "artifact": _artifact(),
            "fallback_policy": "conservative_shrink",
            "min_bucket_samples": 5,
            "min_calibrated_net_edge": 0.09,
            "min_breakeven_margin": 0.09,
        },
    )

    assert result is not None
    assert result.applied is True
    assert result.calibrated_p_side == 0.70
    assert result.rejected is True
    assert result.reason in {"calibrated_net_edge_below_threshold", "calibrated_breakeven_margin_below_threshold"}
