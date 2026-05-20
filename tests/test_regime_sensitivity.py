from src.backtest.regime_sensitivity import (
    DEFAULT_ANCHOR_PARAMS,
    build_quiet_consensus_chase_grid,
    build_quiet_consensus_chase_rule,
    evaluate_rule_grid,
    quiet_rule_name,
    summarize_sensitivity,
    train_test_validate_grid,
)


def _matching_trade(run: str, pnl: float) -> dict:
    return {
        "run": run,
        "pnl": pnl,
        "filled": True,
        "agreement_ratio": 0.90,
        "is_chase": True,
        "spot_return_bps_from_open": 1.4,
        "spot_recent_return_1m_bps": 0.2,
        "spot_recent_vol_5m_bps": 1.8,
    }


def test_quiet_consensus_chase_rule_matches_anchor_regime() -> None:
    rule = build_quiet_consensus_chase_rule(DEFAULT_ANCHOR_PARAMS)
    assert rule.name == quiet_rule_name(DEFAULT_ANCHOR_PARAMS)
    assert rule.matches(_matching_trade("run_a", -3.0)) is True
    assert rule.matches({**_matching_trade("run_a", -3.0), "spot_recent_vol_5m_bps": 2.8}) is False


def test_sensitivity_summary_counts_anchor_and_local_robust_rules() -> None:
    grid = build_quiet_consensus_chase_grid(
        agreement_values=(0.80, 0.85),
        chase_absret_values=(1.0,),
        recent_flat_values=(0.5,),
        vol5m_values=(2.0,),
    )
    trades = [
        _matching_trade("train", -5.0),
        _matching_trade("train", -4.0),
        {**_matching_trade("train", 3.0), "spot_recent_vol_5m_bps": 3.0},
    ]

    summary = summarize_sensitivity(evaluate_rule_grid(trades, grid), min_skipped_filled=1)

    assert summary["anchor_rank"] == 1
    assert summary["robust_rule_count"] == 2
    assert summary["local_robust_rule_count"] == 2


def test_train_test_validate_grid_selects_on_train_and_reports_test_delta() -> None:
    grid = build_quiet_consensus_chase_grid(
        agreement_values=(0.80,),
        chase_absret_values=(1.0,),
        recent_flat_values=(0.5,),
        vol5m_values=(2.0,),
    )
    trades = [
        _matching_trade("train", -5.0),
        _matching_trade("train", -4.0),
        {**_matching_trade("train", 2.0), "spot_recent_return_1m_bps": 1.2},
        _matching_trade("phase7", -3.0),
        {**_matching_trade("phase7", 1.0), "spot_recent_return_1m_bps": 1.2},
    ]

    out = train_test_validate_grid(
        trades,
        grid,
        train_runs=["train"],
        test_runs=["phase7"],
        target_run="phase7",
        min_train_skipped_filled=1,
    )

    assert out["eligible_train_rule_count"] == 1
    assert out["selected_rules"][0]["train"]["delta"] == 9.0
    assert out["selected_rules"][0]["test"]["delta"] == 3.0
