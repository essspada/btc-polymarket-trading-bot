from __future__ import annotations

from scripts.replay_taker_money_machine import DEFAULT_DATASETS, ReplayConfig, Trade, select_trades, simulate_bankroll


def test_default_replay_datasets_point_at_bundled_sample() -> None:
    assert DEFAULT_DATASETS
    for _name, _window, path in DEFAULT_DATASETS:
        assert "data/sample" in str(path) and path.name == "outcomes_5m.jsonl"


def test_replay_liquidity_cap_limits_taker_cash() -> None:
    trade = Trade(
        side="up",
        actual_side="up",
        bid_price=0.59,
        ask_price=0.60,
        gate_price=0.60,
        spread=0.01,
        top3_ask_size=10.0,
        ts="2026-05-06T00:00:00+00:00",
    )
    result = simulate_bankroll(
        [trade],
        100.0,
        ReplayConfig(
            risk_fraction=1.0,
            max_trade_usd=100.0,
            min_trade_usd=0.0,
            fee_rate=0.072,
            fee_source="config",
            min_net_edge=-1.0,
        ),
    )

    assert result.executed_trades == 1
    assert result.liquidity_capped_trades == 1
    assert result.final_balance < 110.0


def test_replay_defaults_match_current_money_machine_strategy() -> None:
    cfg = ReplayConfig()

    assert cfg.gate_price == "ask"
    assert cfg.min_edge == 0.01
    assert cfg.proxy_min_edge == 0.0
    assert cfg.min_net_edge == 0.01
    assert cfg.max_spread == 0.02
    assert cfg.min_price == 0.01
    assert cfg.max_price == 0.70
    assert cfg.fee_source == "row"


def test_replay_row_fee_bps_filters_by_net_edge() -> None:
    row = {
        "actual_side": "up",
        "timing_policy_stage_delay": 240,
        "candidate_models": {
            "spot_logistic_online": {"p_up": 0.535},
            "proxy_logistic_market_blend": {"p_up": 0.535},
        },
        "up_best_bid": 0.50,
        "up_best_ask": 0.51,
        "up_spread": 0.01,
        "up_top3_ask_size": 100.0,
        "up_taker_fee_bps": 1000.0,
    }

    gross_only = ReplayConfig(fee_source="row", min_edge=0.01, proxy_min_edge=0.0, min_net_edge=-1.0)
    net_filtered = ReplayConfig(fee_source="row", min_edge=0.01, proxy_min_edge=0.0, min_net_edge=0.01)

    assert len(select_trades([row], gross_only)) == 1
    assert select_trades([row], net_filtered) == []


def test_replay_default_ask_gate_rejects_bid_only_edge() -> None:
    row = {
        "actual_side": "up",
        "timing_policy_stage_delay": 240,
        "candidate_models": {
            "spot_logistic_online": {"p_up": 0.54},
            "proxy_logistic_market_blend": {"p_up": 0.54},
        },
        "up_best_bid": 0.50,
        "up_best_ask": 0.52,
        "up_spread": 0.02,
        "up_top3_ask_size": 100.0,
    }

    assert select_trades([row], ReplayConfig(gate_price="ask", min_edge=0.03, proxy_min_edge=0.03)) == []
    assert len(select_trades([row], ReplayConfig(gate_price="bid", min_edge=0.03, proxy_min_edge=0.03, min_net_edge=-1.0))) == 1


def test_replay_can_reject_thin_top3_ask_liquidity() -> None:
    row = {
        "actual_side": "up",
        "timing_policy_stage_delay": 240,
        "candidate_models": {
            "spot_logistic_online": {"p_up": 0.62},
            "proxy_logistic_market_blend": {"p_up": 0.60},
        },
        "up_best_bid": 0.49,
        "up_best_ask": 0.50,
        "up_spread": 0.01,
        "up_top3_ask_size": 120.0,
    }

    assert len(select_trades([row], ReplayConfig(min_top3_ask_size=None, min_net_edge=0.0))) == 1
    assert select_trades([row], ReplayConfig(min_top3_ask_size=150.0, min_net_edge=0.0)) == []


def test_replay_actionability_calibration_rejects_row_with_insufficient_bucket_support() -> None:
    row = {
        "actual_side": "up",
        "timing_policy_stage_delay": 240,
        "candidate_models": {
            "spot_logistic_online": {"p_up": 0.62},
            "proxy_logistic_market_blend": {"p_up": 0.62},
        },
        "up_best_bid": 0.49,
        "up_best_ask": 0.50,
        "up_spread": 0.01,
        "up_top3_ask_size": 120.0,
        "up_top3_bid_size": 110.0,
        "spot_recent_vol_5m_bps": 8.0,
    }
    cfg = ReplayConfig(
        min_edge=0.01,
        proxy_min_edge=0.0,
        min_net_edge=0.0,
        actionability_calibration={
            "enabled": True,
            "fallback_policy": "reject",
            "min_bucket_samples": 5,
            "artifact": {
                "schema": "actionability_calibration_v1",
                "global": {"n": 20, "wins": 12, "lb_wr": 0.52},
                "features": {
                    "confidence": {
                        "buckets": [
                            {"lower": 0.50, "upper": 0.60, "n": 10, "wins": 5, "lb_wr": 0.40},
                        ]
                    }
                },
            },
        },
    )

    assert select_trades([row], cfg) == []
