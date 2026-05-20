from src.backtest.redecision_reliability_overlay import synthetic_row_from_redecision


def test_synthetic_row_from_redecision_rewrites_policy_probability_and_trade_result() -> None:
    row = {
        "p_up": 0.9,
        "predicted_side": "up",
        "actual_side": "down",
        "down_best_bid": 0.40,
        "candidate_models": {"a": {"p_up": 0.2}},
    }
    decision = {
        "policy": "spot_consensus_blend",
        "candidate_key": "spot_consensus_blend",
        "p_up": 0.25,
        "action": "trade",
        "reason": "ok",
        "best_edge": 0.11,
        "order_type": "MAKER",
        "selected_side": "down",
        "pnl_if_filled": 4.0,
        "cash_required_if_filled": 10.0,
        "size": 25.0,
    }

    out = synthetic_row_from_redecision(row, decision, fill_mode="assume_filled")
    assert out["p_up"] == 0.25
    assert out["predicted_side"] == "down"
    assert out["confidence"] == 0.75
    assert out["decision_action"] == "trade"
    assert out["trade_filled"] is True
    assert out["trade_net_pnl"] == 4.0
    assert out["source_policy"] == "spot_consensus_blend"


def test_synthetic_row_from_redecision_paper_sim_uses_simulated_fill() -> None:
    row = {"p_up": 0.5, "actual_side": "up"}
    decision = {
        "p_up": 0.7,
        "action": "trade",
        "selected_side": "up",
        "paper_sim_filled": False,
        "paper_sim_pnl": 9.0,
    }
    out = synthetic_row_from_redecision(row, decision, fill_mode="paper_sim")
    assert out["trade_filled"] is False
    assert out["trade_net_pnl"] == 0.0
