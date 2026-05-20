from __future__ import annotations

from scripts.report_timing_policy_nested_oos import select_best_margin_result


def test_select_best_margin_prefers_zero_negative_days_even_if_not_max_balance() -> None:
    margin, result = select_best_margin_result(
        {
            0.08: {"final_balance": 4900.0, "negative_days": 0, "max_drawdown_pct": 0.10},
            0.10: {"final_balance": 5100.0, "negative_days": 1, "max_drawdown_pct": 0.18},
        }
    )
    assert margin == 0.08
    assert result["negative_days"] == 0


def test_select_best_margin_uses_balance_when_negative_days_tie() -> None:
    margin, result = select_best_margin_result(
        {
            0.05: {"final_balance": 4300.0, "negative_days": 0, "max_drawdown_pct": 0.11},
            0.08: {"final_balance": 4700.0, "negative_days": 0, "max_drawdown_pct": 0.13},
        }
    )
    assert margin == 0.08
    assert result["final_balance"] == 4700.0
