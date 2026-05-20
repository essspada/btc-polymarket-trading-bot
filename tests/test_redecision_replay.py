import json

import pytest

from src.backtest.redecision import (
    RedecisionConfig,
    RedecisionPolicy,
    load_rollout_rows,
    redecide_row,
    run_redecision_replay,
)
from src.polymarket.fees import FeeModelConfig
from src.polymarket.risk import RiskManager


def _row(**overrides):
    row = {
        "market_id": "m1",
        "market_slug": "btc-updown-test",
        "created_at": "2026-04-24T03:39:07+00:00",
        "seconds_to_expiry": 60.0,
        "p_up": 0.80,
        "up_token_id": "up",
        "down_token_id": "down",
        "up_best_bid": 0.55,
        "up_best_ask": 0.56,
        "up_midpoint": 0.555,
        "up_spread": 0.01,
        "down_best_bid": 0.43,
        "down_best_ask": 0.44,
        "down_midpoint": 0.435,
        "down_spread": 0.01,
        "up_taker_fee_bps": 0.0,
        "down_taker_fee_bps": 0.0,
        "actual_side": "up",
    }
    row.update(overrides)
    return row


def _exec_cfg(**overrides):
    cfg = {
        "allowed_order_types": ["maker"],
        "maker_preference": True,
        "lock_side_to_prediction": True,
        "min_edge_to_trade": 0.001,
        "min_edge_for_taker": 0.001,
        "min_reward_to_risk_ratio": 0.0,
        "maker_fill_probability": 0.65,
        "maker_fill_floor": 0.05,
        "maker_fill_cap": 0.9,
        "maker_fill_base": 0.78,
        "maker_fill_spread_penalty": 7.0,
        "maker_fill_late_penalty_90": 0.15,
        "maker_fill_late_penalty_45": 0.10,
        "maker_ev_advantage_required": 0.0005,
        "slippage_bps_taker": 0.0,
        "slippage_bps_maker": 0.0,
    }
    cfg.update(overrides)
    return cfg


def _risk_cfg(**overrides):
    cfg = {
        "max_exposure_per_window_usd": 100.0,
        "max_balance_fraction_per_trade": 1.0,
    }
    cfg.update(overrides)
    return cfg


def test_redecide_row_uses_top_level_probability_and_computes_profit_if_filled() -> None:
    out = redecide_row(
        _row(),
        policy=RedecisionPolicy(name="runtime", candidate_key="p_up"),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        balance=100.0,
    )
    assert out["action"] == "trade"
    assert out["order_type"] == "MAKER"
    assert out["selected_side"] == "up"
    assert out["win_if_filled"] is True
    assert out["pnl_if_filled"] > 0.0


def test_redecide_row_can_read_nested_candidate_model_probability() -> None:
    row = _row(
        p_up=0.20,
        candidate_models={"nested_edge": {"p_up": 0.82}},
    )
    out = redecide_row(
        row,
        policy=RedecisionPolicy(name="nested", candidate_key="nested_edge"),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        balance=100.0,
    )
    assert out["action"] == "trade"
    assert out["p_up"] == 0.82
    assert out["selected_side"] == "up"


def test_redecision_policy_overrides_can_block_with_profit_gate() -> None:
    out = redecide_row(
        _row(p_up=0.58),
        policy=RedecisionPolicy(
            name="strict_roi",
            candidate_key="p_up",
            exec_overrides={"min_expected_roi_cash": 0.25},
        ),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        balance=100.0,
    )
    assert out["action"] == "no_trade"
    assert out["reason"] == "expected_roi_cash_below_threshold"


def test_redecision_applies_runtime_sizing_cap_before_pnl() -> None:
    out = redecide_row(
        _row(),
        policy=RedecisionPolicy(name="runtime", candidate_key="p_up"),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(sizing_cap={"enabled": True, "max_trade_usd": 8.0, "min_trade_usd_after_cap": 5.0}),
        balance=100.0,
    )

    assert out["action"] == "trade"
    assert out["decision_cash_required"] == pytest.approx(8.0)
    assert out["cash_required_if_filled"] == pytest.approx(8.0)
    assert out["sizing_cap_payload"]["sizing_cap_applied"] is True
    assert out["runtime_raw_cash_required"] > out["decision_cash_required"]


def test_redecision_applies_active_x3_after_sizing_cap() -> None:
    out = redecide_row(
        _row(),
        policy=RedecisionPolicy(name="runtime", candidate_key="p_up"),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(
            sizing_cap={"enabled": True, "max_trade_usd": 18.0, "min_trade_usd_after_cap": 5.0},
            x3_resolver={
                "enabled": True,
                "mode": "active",
                "policy": "cap_large_cash12",
                "large_cash_threshold_usd": 16.0,
                "large_cash_cap_usd": 12.0,
                "min_trade_usd_after_cap": 5.0,
            },
        ),
        balance=100.0,
    )

    assert out["action"] == "trade"
    assert out["decision_cash_required"] == pytest.approx(12.0)
    assert out["cash_required_if_filled"] == pytest.approx(12.0)
    assert out["x3_payload"]["x3_runtime_applied"] is True
    assert out["x3_payload"]["x3_action"] == "cap"
    assert out["x3_payload"]["x3_candidate_side"] == "up"


def test_redecision_applies_stateful_multiplier_before_sizing_and_x3() -> None:
    stateful_cfg = {
        "enabled": True,
        "drawdown_soft_pct": 0.01,
        "drawdown_hard_pct": 0.05,
        "daily_loss_soft_frac": 0.01,
        "daily_loss_hard_frac": 0.05,
        "loss_streak_soft": 1,
        "loss_streak_hard": 2,
        "min_multiplier": 0.50,
        "recovery_wins_required": 0,
    }
    stateful_risk = RiskManager(
        max_exposure_per_window_usd=100.0,
        max_daily_loss_usd=100.0,
        cooldown_after_loss_streak=999,
        cooldown_windows=0,
        max_open_positions=1,
        stateful_regime=stateful_cfg,
    )
    stateful_risk.on_trade_close(-10.0)

    out = redecide_row(
        _row(),
        policy=RedecisionPolicy(name="runtime", candidate_key="p_up"),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(
            sizing_cap={"enabled": True, "max_trade_usd": 18.0, "min_trade_usd_after_cap": 5.0},
            stateful_regime=stateful_cfg,
        ),
        balance=90.0,
        peak_balance=100.0,
        stateful_risk=stateful_risk,
    )

    assert out["action"] == "trade"
    assert out["stateful_payload"]["stateful_multiplier_applied"] is True
    assert out["stateful_payload"]["stateful_reason"] in {"stateful_hard_throttle", "stateful_soft_throttle"}
    assert out["stateful_payload"]["stateful_capped_cash_required"] < out["runtime_raw_cash_required"]
    assert out["decision_cash_required"] <= out["stateful_payload"]["stateful_capped_cash_required"]


def test_run_redecision_replay_compares_policies() -> None:
    rows = [
        _row(market_id="m1", actual_side="up", p_up=0.80),
        _row(market_id="m2", actual_side="down", p_up=0.80),
    ]
    result = run_redecision_replay(
        rows,
        policies=(
            RedecisionPolicy(name="runtime", candidate_key="p_up"),
            RedecisionPolicy(name="blocked", candidate_key="p_up", exec_overrides={"min_expected_roi_cash": 0.99}),
        ),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        config=RedecisionConfig(initial_balance=100.0, include_decisions=True),
    )
    runtime = result["policies"]["runtime"]
    blocked = result["policies"]["blocked"]
    assert runtime["trades"] == 2
    assert runtime["wins"] == 1
    assert runtime["losses"] == 1
    assert blocked["trades"] == 0
    assert blocked["skip_or_reason_counts"]["expected_roi_cash_below_threshold"] == 2
    assert len(runtime["decisions"]) == 2


def test_run_redecision_replay_carries_stateful_state_across_rows() -> None:
    rows = [
        _row(market_id="m1", actual_side="down", p_up=0.80, created_at="2026-04-24T03:39:07+00:00"),
        _row(market_id="m2", actual_side="up", p_up=0.80, created_at="2026-04-24T03:44:07+00:00"),
    ]
    result = run_redecision_replay(
        rows,
        policies=(RedecisionPolicy(name="runtime", candidate_key="p_up"),),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(
            stateful_regime={
                "enabled": True,
                "drawdown_soft_pct": 0.01,
                "drawdown_hard_pct": 0.02,
                "daily_loss_soft_frac": 0.01,
                "daily_loss_hard_frac": 0.02,
                "loss_streak_soft": 1,
                "loss_streak_hard": 1,
                "min_multiplier": 0.40,
                "recovery_wins_required": 0,
            }
        ),
        config=RedecisionConfig(initial_balance=100.0, include_decisions=True),
    )

    decisions = result["policies"]["runtime"]["decisions"]
    assert decisions[0]["stateful_payload"]["stateful_reason"] == "stateful_ok"
    assert decisions[1]["stateful_payload"]["stateful_multiplier_applied"] is True
    assert decisions[1]["decision_cash_required"] < decisions[1]["runtime_raw_cash_required"]


def test_run_redecision_replay_applies_runtime_cooldown_gate() -> None:
    rows = [
        _row(market_id="m1", actual_side="down", p_up=0.80, created_at="2026-04-24T03:39:07+00:00"),
        _row(market_id="m2", actual_side="up", p_up=0.80, created_at="2026-04-24T03:40:07+00:00"),
    ]
    result = run_redecision_replay(
        rows,
        policies=(RedecisionPolicy(name="runtime", candidate_key="p_up"),),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(
            cooldown_after_loss_streak=1,
            cooldown_windows=2,
            max_open_positions=1,
        ),
        config=RedecisionConfig(initial_balance=100.0, include_decisions=True),
    )

    runtime = result["policies"]["runtime"]
    assert runtime["trades"] == 1
    assert runtime["decisions"][1]["action"] == "no_trade"
    assert runtime["decisions"][1]["reason"] == "cooldown_or_drift"


def test_run_redecision_replay_can_use_paper_fill_simulation() -> None:
    result = run_redecision_replay(
        [_row(market_id="m1", actual_side="up", p_up=0.80)],
        policies=(RedecisionPolicy(name="runtime", candidate_key="p_up"),),
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        exec_cfg=_exec_cfg(allowed_order_types=["taker"]),
        risk_cfg=_risk_cfg(),
        config=RedecisionConfig(initial_balance=100.0, fill_mode="paper_sim", assume_filled=False),
    )
    runtime = result["policies"]["runtime"]
    assert runtime["fill_mode"] == "paper_sim"
    assert runtime["submitted_trades"] == 1
    assert runtime["filled_trades"] == 1
    assert runtime["net_pnl"] > 0.0


def test_load_rollout_rows_merges_token_ids_from_trade_log(tmp_path) -> None:
    (tmp_path / "paper_outcomes_5m.jsonl").write_text(
        json.dumps({"market_id": "m1", "p_up": 0.8}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "paper_trades_5m.jsonl").write_text(
        json.dumps({"market_id": "m1", "up_token_id": "up-real", "down_token_id": "down-real"}) + "\n",
        encoding="utf-8",
    )
    rows = load_rollout_rows(tmp_path)
    assert rows[0]["up_token_id"] == "up-real"
    assert rows[0]["down_token_id"] == "down-real"
