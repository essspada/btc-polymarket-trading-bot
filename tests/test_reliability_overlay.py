from src.backtest.reliability_overlay import evaluate_reliability_overlay
from src.strategy.reliability_memory import build_reliability_memory


def _row(pnl: float, *, up_bid: float = 0.62) -> dict:
    return {
        "predicted_side": "up",
        "actual_side": "down" if pnl < 0 else "up",
        "confidence": 0.91,
        "decision_action": "trade",
        "decision_order_type": "MAKER",
        "decision_best_edge": 0.08,
        "timing_policy_reason": "edge_hit",
        "spot_recent_vol_5m_bps": 3.0,
        "up_best_bid": up_bid,
        "trade_filled": True,
        "trade_net_pnl": pnl,
        "sizing_cap_capped_cash_required": 10.0,
        "candidate_models": {
            "a": {"p_up": 0.20},
            "b": {"p_up": 0.30},
            "c": {"p_up": 0.40},
        },
    }


def test_reliability_overlay_skips_toxic_veto_and_saves_loss() -> None:
    memory = build_reliability_memory([_row(-4.0), _row(-5.0), _row(-3.0)], prior_filled=0.0)
    out = evaluate_reliability_overlay([_row(-2.0)], memory, max_shrunk_roi=-0.05, min_entry_filled=3)
    assert out["original_pnl"] == -2.0
    assert out["overlay_pnl"] == 0.0
    assert out["delta_pnl"] == 2.0
    assert out["saved_losses"] == 2.0


def test_reliability_overlay_counts_missed_profit_when_toxic_rule_over_skips() -> None:
    memory = build_reliability_memory([_row(-4.0), _row(-5.0), _row(-3.0)], prior_filled=0.0)
    out = evaluate_reliability_overlay([_row(6.0)], memory, max_shrunk_roi=-0.05, min_entry_filled=3)
    assert out["original_pnl"] == 6.0
    assert out["overlay_pnl"] == 0.0
    assert out["delta_pnl"] == -6.0
    assert out["missed_profit"] == 6.0


def test_reliability_overlay_can_block_an_expert_from_actionable_vetoes() -> None:
    memory = build_reliability_memory([_row(-4.0), _row(-5.0), _row(-3.0)], prior_filled=0.0)
    out = evaluate_reliability_overlay(
        [_row(-2.0)],
        memory,
        max_shrunk_roi=-0.05,
        min_entry_filled=3,
        blocked_experts={"candidate_agreement_expert"},
    )
    assert out["overlay_pnl"] == -2.0
    assert out["skipped_filled"] == 0
