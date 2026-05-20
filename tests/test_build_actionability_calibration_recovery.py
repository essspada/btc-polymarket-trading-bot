from __future__ import annotations

from scripts.build_actionability_calibration_recovery import _gate, build_artifact


def test_build_artifact_contains_runtime_fields_only_metadata() -> None:
    artifact = build_artifact(
        train_samples=[
            {
                "win": 1,
                "confidence": 0.65,
                "selected_price": 0.52,
                "selected_spread": 0.01,
                "selected_top3_ask_size": 500.0,
                "model_proxy_gap": 0.01,
                "decision_breakeven_margin": 0.04,
                "spot_recent_vol_5m_bps": 8.0,
            }
        ],
        min_bucket_samples=5,
        conservative_alpha=0.7,
        z_score=1.28155,
    )

    assert artifact["schema"] == "actionability_calibration_v1"
    assert "runtime_fields_only" in artifact
    assert "actual_side" not in str(artifact)


def test_gate_marks_candidate_non_promotable_when_primary_holdout_is_negative() -> None:
    payload = {
        "historical": {
            "return_pct_1000": 10.0,
            "profit_factor": 1.2,
            "selected_wr_pct": 60.0,
            "breakeven_wr_pct": 45.0,
            "max_drawdown_pct_1000": 10.0,
            "pnl_without_top_winner": 1.0,
            "monotonic_edge_ranking_ok": True,
            "selected_trades": 20,
        },
        "may6": {
            "return_pct_1000": 3.0,
            "profit_factor": 1.12,
            "selected_wr_pct": 58.0,
            "breakeven_wr_pct": 50.0,
            "max_drawdown_pct_1000": 12.0,
            "pnl_without_top_winner": 0.5,
            "monotonic_edge_ranking_ok": True,
            "selected_trades": 10,
        },
        "may8": {
            "return_pct_1000": -1.0,
            "profit_factor": 0.95,
            "selected_wr_pct": 45.0,
            "breakeven_wr_pct": 49.0,
            "max_drawdown_pct_1000": 14.0,
            "pnl_without_top_winner": -0.2,
            "monotonic_edge_ranking_ok": False,
            "selected_trades": 11,
        },
        "liq150_stopped": {
            "return_pct_1000": 1.0,
            "profit_factor": 1.01,
            "selected_wr_pct": 60.0,
            "breakeven_wr_pct": 50.0,
            "max_drawdown_pct_1000": 8.0,
            "pnl_without_top_winner": 0.1,
            "monotonic_edge_ranking_ok": True,
            "selected_trades": 4,
        },
    }
    verdict = _gate(payload)

    assert verdict["promotable"] is False
    assert any(item.startswith("may8:") for item in verdict["reasons"])
