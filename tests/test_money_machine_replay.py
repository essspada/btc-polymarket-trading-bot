from scripts.replay_money_machine import render_markdown
from src.backtest.money_machine import (
    SizingSpec,
    StrategySpec,
    build_replay_report,
    empirical_p_up_series,
    fixed_cash_metrics,
    select_trade,
    simulate_bankroll,
)


def _row(**overrides):
    row = {
        "created_at": "2026-05-05T00:00:00+00:00",
        "actual_side": "down",
        "predicted_side": "up",
        "timing_policy_stage_delay": 240,
        "up_best_bid": 0.50,
        "up_spread": 0.01,
        "down_best_bid": 0.49,
        "down_spread": 0.01,
        "candidate_models": {
            "spot_logistic_online": {"p_up": 0.56},
            "proxy_logistic_market_blend": {"p_up": 0.55},
        },
    }
    row.update(overrides)
    return row


def test_fixed_cash_metrics_uses_actual_side_not_predicted_side() -> None:
    strategy = StrategySpec(name="test", source_key="spot_logistic_online", min_edge=0.03)
    metrics = fixed_cash_metrics([_row()], strategy, cash=100.0)
    assert metrics["trades"] == 1
    assert metrics["wins"] == 0
    assert metrics["pnl"] == -100.0


def test_proxy_confirm_strategy_requires_same_side_proxy_edge() -> None:
    strategy = StrategySpec(
        name="test",
        source_key="spot_logistic_online",
        min_edge=0.03,
        confirm_key="proxy_logistic_market_blend",
        confirm_min_edge=0.03,
    )
    assert select_trade(_row(), strategy) is not None
    assert select_trade(_row(candidate_models={"spot_logistic_online": {"p_up": 0.56}, "proxy_logistic_market_blend": {"p_up": 0.45}}), strategy) is None
    assert select_trade(_row(candidate_models={"spot_logistic_online": {"p_up": 0.56}, "proxy_logistic_market_blend": {"p_up": 0.52}}), strategy) is None


def test_strategy_spec_can_use_ask_entry_price_for_taker_style_replay() -> None:
    row = _row(
        actual_side="up",
        up_best_bid=0.50,
        up_best_ask=0.52,
        candidate_models={
            "spot_logistic_online": {"p_up": 0.515},
            "proxy_logistic_market_blend": {"p_up": 0.515},
        },
    )

    bid_strategy = StrategySpec(name="bid", source_key="spot_logistic_online", min_edge=0.01, entry_price="bid")
    ask_strategy = StrategySpec(name="ask", source_key="spot_logistic_online", min_edge=0.01, entry_price="ask")

    bid_decision = select_trade(row, bid_strategy)
    assert bid_decision is not None
    assert bid_decision.price == 0.50
    assert select_trade(row, ask_strategy) is None


def test_simulate_bankroll_can_use_row_level_taker_fee_bps() -> None:
    row = _row(
        actual_side="up",
        up_best_bid=0.49,
        up_best_ask=0.50,
        up_taker_fee_bps=1000.0,
        candidate_models={
            "spot_logistic_online": {"p_up": 0.70},
            "proxy_logistic_market_blend": {"p_up": 0.70},
        },
    )
    strategy = StrategySpec(name="taker", source_key="spot_logistic_online", min_edge=0.01, entry_price="ask")
    sizing = SizingSpec(name="flat", mode="flat_fraction", risk_fraction=1.0, max_trade_usd=100.0, min_trade_usd=0.0)

    config_fee = simulate_bankroll([row], strategy, sizing, initial_balance=100.0, fee_rate=0.0)
    row_fee = simulate_bankroll(
        [row],
        strategy,
        sizing,
        initial_balance=100.0,
        fee_rate=0.0,
        fee_rate_source="row",
    )

    assert config_fee["final_balance"] == 200.0
    assert row_fee["final_balance"] == 195.0
    assert row_fee["final_balance"] < config_fee["final_balance"]


def test_empirical_series_uses_only_prior_rows() -> None:
    rows = [
        _row(created_at=f"2026-05-05T00:0{i}:00+00:00", actual_side="up" if i < 3 else "down")
        for i in range(5)
    ]
    probs = empirical_p_up_series(rows, warmup_rows=3, nearest_k=4, prior_weight=0.0)
    assert probs[:3] == [None, None, None]
    assert probs[3] == 1.0
    assert probs[4] < 1.0


def test_build_replay_report_marks_legacy_non_runtime_parity() -> None:
    payload = build_replay_report([_row(actual_side="up", predicted_side="up", up_best_ask=0.51)])
    assumptions = payload["assumptions"]
    assert assumptions["runtime_parity"] is False
    assert assumptions["replay_mode"] == "legacy_bid_maker_independent_fill_research"
    assert "not runtime-equivalent" in assumptions["warning"]


def test_render_markdown_includes_legacy_warning() -> None:
    payload = build_replay_report([_row(actual_side="up", predicted_side="up", up_best_ask=0.51)])
    text = render_markdown(payload)
    assert "legacy research replay" in text
    assert "not runtime-parity forward evidence" in text
