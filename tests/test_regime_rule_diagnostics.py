from src.backtest.regime_rule_diagnostics import (
    RuleClause,
    RuleSpec,
    build_phase7_rule_specs,
    build_trade_features,
    evaluate_skip_rule,
    rank_rule_results,
)


def test_build_trade_features_adds_regime_fields() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_p_up": 0.60,
            "seconds_to_expiry": 55,
            "spot_return_bps_from_open": 2.5,
            "spot_recent_return_1m_bps": -0.1,
            "spot_recent_vol_5m_bps": 1.8,
            "candidate_models": {
                "a": {"p_up": 0.80},
                "b": {"p_up": 0.75},
                "c": {"p_up": 0.40},
            },
        }
    ]
    decisions = [
        {
            "market_id": "m1",
            "action": "trade",
            "selected_side": "up",
            "actual_side": "down",
            "p_up": 0.82,
            "price": 0.62,
            "expected_roi_cash": 0.08,
            "breakeven_margin": 0.10,
            "pnl_if_filled": -6.2,
            "win_if_filled": False,
        }
    ]

    trade = build_trade_features(rows, decisions, run_name="run_a")[0]

    assert trade["is_chase"] is True
    assert trade["recent_opposes_side"] is True
    assert trade["agreement_ratio"] == 2 / 3
    assert round(trade["model_market_gap"], 3) == 0.22


def test_build_trade_features_can_use_paper_sim_fill_mode() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_p_up": 0.60,
            "spot_return_bps_from_open": 2.5,
            "spot_recent_return_1m_bps": 0.1,
            "spot_recent_vol_5m_bps": 1.8,
        }
    ]
    decisions = [
        {
            "market_id": "m1",
            "action": "trade",
            "selected_side": "up",
            "p_up": 0.82,
            "pnl_if_filled": -6.2,
            "win_if_filled": False,
            "paper_sim_filled": False,
            "paper_sim_pnl": -6.2,
        }
    ]

    trade = build_trade_features(rows, decisions, run_name="run_a", fill_mode="paper_sim")[0]

    assert trade["filled"] is False
    assert trade["pnl"] == 0.0


def test_evaluate_skip_rule_saves_losses_and_counts_missed_profit() -> None:
    rule = RuleSpec("toxic", (RuleClause("match", lambda trade: bool(trade["match"])),))
    trades = [
        {"run": "a", "pnl": -5.0, "match": True},
        {"run": "a", "pnl": 2.0, "match": True},
        {"run": "a", "pnl": 3.0, "match": False},
        {"run": "b", "pnl": 4.0, "match": False},
    ]

    out = evaluate_skip_rule(trades, rule)

    assert out["delta"] == 3.0
    assert out["saved_losses"] == 5.0
    assert out["missed_profit"] == 2.0
    assert out["negative_delta_runs"] == 0
    assert out["per_run"][0]["overlay_pnl"] == 3.0


def test_phase7_specs_are_bounded_and_rank_target_run_first() -> None:
    specs = build_phase7_rule_specs()
    assert specs
    assert len(specs) < 500

    ranked = rank_rule_results(
        [
            {"rule": "bad", "delta": 10.0, "negative_delta_runs": 0, "skipped_trades": 1, "per_run": [{"run": "phase7", "overlay_pnl": -1.0, "delta": 5.0}]},
            {"rule": "good", "delta": 9.0, "negative_delta_runs": 0, "skipped_trades": 1, "per_run": [{"run": "phase7", "overlay_pnl": 1.0, "delta": 4.0}]},
        ],
        target_run="phase7",
    )

    assert ranked[0]["rule"] == "good"
