from scripts.report_same_window_candidates import _meta_policy_candidate


def test_meta_policy_prefers_logistic_when_proxy_rules_do_not_fire() -> None:
    candidates = {
        "proxy_orderbook": {
            "p_up": 0.62,
            "predicted_side": "up",
            "simulated_trade": True,
            "simulated_selected_side": "up",
        },
        "spot_logistic_online": {
            "p_up": 0.31,
            "predicted_side": "down",
            "simulated_trade": True,
            "simulated_selected_side": "down",
        },
        "spot_logistic_market_blend": {
            "p_up": 0.48,
            "predicted_side": "down",
            "simulated_trade": True,
            "simulated_selected_side": "up",
        },
    }

    out = _meta_policy_candidate(
        candidates,
        proxy_midpoint_band=0.06,
        proxy_edge_margin=0.0,
        fallback_source="spot_logistic_online",
    )

    assert out is not None
    assert out["p_up"] == candidates["spot_logistic_online"]["p_up"]
    assert out["meta_source"] == "fallback"
    assert out["meta_reason"] == "fallback_candidate"


def test_meta_policy_prefers_logistic_when_selected_side_agrees() -> None:
    candidates = {
        "proxy_orderbook": {
            "p_up": 0.56,
            "predicted_side": "up",
            "simulated_trade": True,
            "simulated_selected_side": "down",
            "decision_expected_edge": 0.05,
        },
        "spot_logistic_online": {
            "p_up": 0.28,
            "predicted_side": "down",
            "simulated_trade": True,
            "simulated_selected_side": "down",
            "decision_expected_edge": 0.12,
        },
    }

    out = _meta_policy_candidate(
        candidates,
        proxy_midpoint_band=0.06,
        proxy_edge_margin=0.0,
        fallback_source="spot_logistic_online",
    )

    assert out is not None
    assert out["p_up"] == candidates["spot_logistic_online"]["p_up"]
    assert out["meta_source"] == "spot_logistic_online"
    assert out["meta_reason"] == "proxy_logistic_agree"


def test_meta_policy_near_midpoint_requires_edge_advantage() -> None:
    candidates = {
        "proxy_orderbook": {
            "p_up": 0.52,
            "predicted_side": "up",
            "simulated_trade": True,
            "simulated_selected_side": "up",
            "decision_expected_edge": 0.03,
        },
        "spot_logistic_online": {
            "p_up": 0.20,
            "predicted_side": "down",
            "simulated_trade": True,
            "simulated_selected_side": "down",
            "decision_expected_edge": 0.10,
        },
    }

    out = _meta_policy_candidate(
        candidates,
        proxy_midpoint_band=0.06,
        proxy_edge_margin=0.0,
        fallback_source="spot_logistic_online",
    )

    assert out is not None
    assert out["p_up"] == candidates["spot_logistic_online"]["p_up"]
    assert out["meta_source"] == "fallback"
    assert out["meta_reason"] == "fallback_candidate"
