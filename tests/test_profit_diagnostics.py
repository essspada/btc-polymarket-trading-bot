from src.backtest.profit_diagnostics import compute_profit_diagnostics


def test_profit_diagnostics_separates_filled_and_unfilled_signal_quality() -> None:
    rows = [
        {
            "predicted_side": "up",
            "actual_side": "up",
            "confidence": 0.91,
            "decision_action": "trade",
            "decision_reason": "ok",
            "decision_best_edge": 0.05,
            "decision_order_type": "MAKER",
            "trade_filled": True,
            "trade_net_pnl": 2.0,
            "up_best_bid": 0.55,
        },
        {
            "predicted_side": "up",
            "actual_side": "down",
            "confidence": 0.92,
            "decision_action": "trade",
            "decision_reason": "ok",
            "decision_best_edge": 0.06,
            "decision_order_type": "MAKER",
            "trade_filled": True,
            "trade_net_pnl": -3.0,
            "up_best_bid": 0.56,
        },
        {
            "predicted_side": "down",
            "actual_side": "down",
            "confidence": 0.75,
            "decision_action": "trade",
            "decision_reason": "ok",
            "decision_best_edge": 0.04,
            "decision_order_type": "MAKER",
            "trade_filled": False,
            "trade_net_pnl": 0.0,
            "down_best_bid": 0.42,
        },
        {
            "predicted_side": "up",
            "actual_side": "up",
            "confidence": 0.65,
            "decision_action": "no_trade",
            "decision_reason": "edge_below_threshold",
            "decision_best_edge": -0.01,
            "trade_filled": False,
            "trade_net_pnl": 0.0,
        },
    ]
    out = compute_profit_diagnostics(rows)
    assert out["overall"]["signals"] == 4
    assert out["overall"]["signal_accuracy"] == 0.75
    assert out["proposed_filled"]["filled"] == 2
    assert out["proposed_filled"]["filled_win_rate"] == 0.5
    assert out["proposed_unfilled"]["signal_accuracy"] == 1.0
    assert out["adverse_selection_signal_accuracy_gap_filled_minus_unfilled"] == -0.5
    assert out["groups"]["by_confidence_bin"]["0.90-0.95"]["pnl"] == -1.0
