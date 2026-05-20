from src.backtest.fill_execution_audit import build_policy_row_audit, summarize_audit_rows, summarize_by_field


def test_build_policy_row_audit_compares_runtime_and_policy_fills() -> None:
    rows = [
        {
            "market_id": "m1",
            "actual_side": "up",
            "predicted_side": "up",
            "decision_action": "trade",
            "trade_filled": True,
            "trade_net_pnl": 4.0,
            "seconds_to_expiry": 120,
        }
    ]
    decisions = [
        {
            "market_id": "m1",
            "action": "trade",
            "selected_side": "up",
            "actual_side": "up",
            "price": 0.60,
            "paper_sim_filled": False,
            "paper_sim_pnl": 0.0,
            "paper_sim_fill_probability": 0.55,
            "pnl_if_filled": 4.0,
        }
    ]

    audit = build_policy_row_audit(rows, decisions, run_name="run", policy_name="policy")
    summary = summarize_audit_rows(audit)

    assert audit[0]["original_filled"] is True
    assert audit[0]["policy_filled"] is False
    assert summary["fill_mismatch_both_trade"] == 1
    assert summary["policy_assume_pnl"] == 4.0
    assert summary["policy_paper_pnl"] == 0.0


def test_summarize_by_field_keeps_policy_pnl_per_bucket() -> None:
    rows = [
        {"price_bucket": "<0.25", "policy_trade": True, "policy_filled": True, "policy_paper_pnl": 3.0},
        {"price_bucket": "<0.25", "policy_trade": True, "policy_filled": True, "policy_paper_pnl": -1.0},
        {"price_bucket": "0.25-0.50", "policy_trade": False, "policy_filled": False, "policy_paper_pnl": 0.0},
    ]

    buckets = summarize_by_field(rows, "price_bucket")

    first = next(item for item in buckets if item["price_bucket"] == "<0.25")
    assert first["policy_trades"] == 2
    assert first["policy_paper_pnl"] == 2.0
