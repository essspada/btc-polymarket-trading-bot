from src.strategy.decision_brain import build_shadow_expert_signals, summarize_shadow_vetoes


def test_shadow_experts_emit_candidate_agreement_and_entry_price_veto() -> None:
    row = {
        "predicted_side": "up",
        "p_up": 0.95,
        "up_best_bid": 0.95,
        "decision_order_type": "MAKER",
        "decision_best_edge": 0.10,
        "book_quality_ok": True,
        "book_quality_score": 1.0,
        "seconds_to_expiry": 60.0,
        "candidate_models": {
            "a": {"p_up": 0.90},
            "b": {"p_up": 0.20},
        },
    }
    signals = {signal.name: signal for signal in build_shadow_expert_signals(row)}
    assert signals["candidate_agreement_expert"].value == 0.5
    assert signals["entry_price_risk_expert"].veto is True
    assert signals["entry_price_risk_expert"].veto_reason == "extreme_entry_price_shadow"


def test_summarize_shadow_vetoes_tracks_pnl_on_veto_rows() -> None:
    rows = [
        {
            "predicted_side": "up",
            "up_best_bid": 0.95,
            "decision_order_type": "MAKER",
            "decision_best_edge": 0.10,
            "trade_filled": True,
            "trade_net_pnl": -5.0,
        }
    ]
    summary = summarize_shadow_vetoes(rows)
    assert summary["entry_price_risk_expert"]["veto_rows"] == 1
    assert summary["entry_price_risk_expert"]["pnl_on_veto_rows"] == -5.0


def test_late_chase_reversal_expert_flags_late_expensive_direction_chase() -> None:
    row = {
        "predicted_side": "up",
        "up_best_bid": 0.72,
        "decision_order_type": "MAKER",
        "decision_best_edge": 0.08,
        "seconds_to_expiry": 55.0,
        "spot_return_bps_from_open": 4.0,
    }
    signals = {signal.name: signal for signal in build_shadow_expert_signals(row)}
    assert signals["late_chase_reversal_expert"].veto is True
    assert signals["late_chase_reversal_expert"].veto_reason == "late_chase_reversal_shadow"


def test_quiet_consensus_chase_expert_flags_low_vol_flat_chase() -> None:
    row = {
        "predicted_side": "down",
        "down_best_bid": 0.68,
        "decision_order_type": "MAKER",
        "decision_best_edge": 0.08,
        "seconds_to_expiry": 55.0,
        "spot_return_bps_from_open": -2.2,
        "spot_recent_return_1m_bps": -0.1,
        "spot_recent_vol_5m_bps": 1.6,
        "candidate_models": {
            "a": {"p_up": 0.20},
            "b": {"p_up": 0.25},
            "c": {"p_up": 0.30},
            "d": {"p_up": 0.35},
            "e": {"p_up": 0.60},
        },
    }
    signals = {signal.name: signal for signal in build_shadow_expert_signals(row)}
    assert signals["quiet_consensus_chase_expert"].veto is True
    assert signals["quiet_consensus_chase_expert"].veto_reason == "quiet_consensus_chase_shadow"
