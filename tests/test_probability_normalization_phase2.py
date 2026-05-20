import json

import numpy as np

from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.runtime import pricing as pricing_mod, timing as timing_mod, utilities as utilities_mod
from src.strategy.calibration import calibrate_probability, clip_prob
from src.strategy.signals import decide_trade
from src.strategy.timing_policy import TimingPolicyStage


class _LinearCalibrator:
    def predict(self, scores: np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float)
        return (0.6 * x) + 0.15


def _books() -> MarketBooks:
    return MarketBooks(
        up=OutcomeBook("up", 0.48, 0.49, 0.485, 0.01, 0.0),
        down=OutcomeBook("down", 0.51, 0.52, 0.515, 0.01, 0.0),
    )


def _exec_cfg() -> dict:
    return {
        "allowed_order_types": ["maker"],
        "maker_preference": True,
        "min_edge_to_trade": 0.001,
        "min_edge_for_taker": 0.001,
        "slippage_bps_taker": 0.0,
        "slippage_bps_maker": 0.0,
        "maker_fill_probability": 1.0,
        "maker_ev_advantage_required": 0.0,
        "lock_side_to_prediction": True,
        "min_reward_to_risk_ratio": 0.0,
    }


def _risk_cfg() -> dict:
    return {"max_exposure_per_window_usd": 100.0}


def test_candidate_payload_uses_same_normalization_formula_and_keeps_raw_fields() -> None:
    calibrator = _LinearCalibrator()
    calibration_shift = 0.07
    raw_p = 0.52
    books = _books()
    fee_cfg = FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0)

    payload = pricing_mod.build_candidate_models_payload(
        candidate_probabilities={"spot_logistic_online": raw_p},
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        fee_bps_map={},
        platt_calibrator=calibrator,
        calibration_shift=calibration_shift,
    )
    item = payload["spot_logistic_online"]

    expected = pricing_mod.normalize_p_up(
        raw_p,
        platt_calibrator=calibrator,
        calibration_shift=calibration_shift,
    )

    assert item["p_up_raw"] == expected["p_up_raw"]
    assert item["p_up_pre_calibration"] == expected["p_up_pre_calibration"]
    assert item["p_up_calibrated"] == expected["p_up_calibrated"]
    assert item["p_up"] == expected["p_up"]
    assert item["is_normalized"] is True

    expected_decision = decide_trade(
        market_slug="m",
        books=books,
        p_up=float(expected["p_up"]),
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.001,
        max_exposure_usd=100.0,
        maker_preference=True,
        fee_cfg=fee_cfg,
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        taker_fee_bps_by_token={},
        maker_fill_probability=float(_exec_cfg()["maker_fill_probability"]),
        maker_fill_probability_by_token=utilities_mod.maker_fill_probability_by_token(
            books=books,
            seconds_to_expiry=240.0,
            exec_cfg=_exec_cfg(),
        ),
        maker_ev_advantage_required=0.0,
        allowed_order_types=["maker"],
        lock_side_to_prediction=True,
        min_reward_to_risk_ratio=0.0,
    )
    assert expected_decision.intent is not None
    assert item["decision_expected_edge"] == expected_decision.intent.expected_edge


def test_timing_finalize_uses_normalized_candidate_without_losing_raw_probability() -> None:
    probs = {
        "p_up": 0.40,
        "p_down": 0.60,
        "predicted_side": "down",
        "confidence": 0.60,
        "p_pre_calibration": 0.40,
        "model_used": "spot_consensus_blend",
        "model_p_up": 0.40,
        "candidate_probabilities": {"spot_logistic_online": 0.40},
    }
    candidate_payload = {
        "spot_logistic_online": {
            "p_up_raw": 0.61,
            "p_up_pre_calibration": 0.61,
            "p_up_calibrated": 0.68,
            "p_up": 0.73,
            "decision_action": "trade",
            "decision_expected_edge": 0.03,
            "predicted_side": "up",
            "confidence": 0.73,
        }
    }
    stages = (TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),)

    action, _selection, updated, _payload = timing_mod.apply_runtime_timing_policy(
        model_source="log240_only",
        timing_policy_stages=stages,
        seconds_from_start=240.0,
        min_edge_to_trade=0.004,
        candidate_models_payload=candidate_payload,
        probs=probs,
        current_stage_index=None,
        late_start_grace_seconds=0.0,
    )

    assert action == "finalize"
    assert updated["p_up"] == 0.73
    assert updated["model_p_up"] == 0.61
    assert updated["p_pre_calibration"] == 0.61
    assert updated["candidate_probabilities"]["log240_only"]["p_up"] == 0.73
    assert updated["candidate_probabilities"]["log240_only"]["is_normalized"] is True


def test_meta_policy_override_writes_normalized_candidate_marker() -> None:
    probs = {
        "p_up": 0.40,
        "p_down": 0.60,
        "predicted_side": "down",
        "confidence": 0.60,
        "p_pre_calibration": 0.40,
        "model_used": "spot_consensus_blend",
        "model_p_up": 0.40,
        "candidate_probabilities": {"spot_logistic_online": 0.40},
    }
    candidate_payload = {
        "proxy_orderbook": {
            "p_up_raw": 0.58,
            "p_up_pre_calibration": 0.58,
            "p_up_calibrated": 0.58,
            "p_up": 0.58,
            "decision_action": "trade",
            "decision_expected_edge": 0.04,
            "decision_token_id": "up",
            "predicted_side": "up",
        },
        "spot_logistic_online": {
            "p_up_raw": 0.62,
            "p_up_pre_calibration": 0.62,
            "p_up_calibrated": 0.62,
            "p_up": 0.62,
            "decision_action": "trade",
            "decision_expected_edge": 0.04,
            "decision_token_id": "up",
            "predicted_side": "up",
        },
    }
    books = _books()

    updated, payload = pricing_mod.apply_meta_policy_override(
        probs=probs,
        candidate_models_payload=candidate_payload,
        books=books,
        model_source="proxy_logistic_meta_policy",
        meta_cfg={},
    )

    assert payload["proxy_logistic_meta_policy"]["p_up"] == updated["p_up"]
    assert updated["candidate_probabilities"]["proxy_logistic_meta_policy"]["p_up"] == updated["p_up"]
    assert updated["candidate_probabilities"]["proxy_logistic_meta_policy"]["is_normalized"] is True


def test_calibration_shift_changes_candidate_probability_and_candidate_edge() -> None:
    raw_p = 0.52
    books = _books()
    fee_cfg = FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0)

    payload_no_shift = pricing_mod.build_candidate_models_payload(
        candidate_probabilities={"spot_logistic_online": raw_p},
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        fee_bps_map={},
        platt_calibrator=None,
        calibration_shift=0.0,
    )
    payload_shifted = pricing_mod.build_candidate_models_payload(
        candidate_probabilities={"spot_logistic_online": raw_p},
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        fee_bps_map={},
        platt_calibrator=None,
        calibration_shift=0.10,
    )

    a = payload_no_shift["spot_logistic_online"]
    b = payload_shifted["spot_logistic_online"]
    assert b["p_up"] > a["p_up"]
    assert b["decision_expected_edge"] > a["decision_expected_edge"]


def test_candidate_normalization_formula_matches_base_formula() -> None:
    calibrator = _LinearCalibrator()
    raw_p = 0.33
    shift = 0.04

    expected = clip_prob(calibrate_probability(raw_p, calibrator) + shift)
    normalized = pricing_mod.normalize_p_up(
        raw_p,
        platt_calibrator=calibrator,
        calibration_shift=shift,
    )
    assert normalized["p_up"] == expected


def test_candidate_payload_is_idempotent_for_already_normalized_input() -> None:
    calibrator = _LinearCalibrator()
    raw_p = 0.53
    books = _books()
    fee_cfg = FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0)
    exec_cfg = _exec_cfg()
    risk_cfg = _risk_cfg()

    first = pricing_mod.build_candidate_models_payload(
        candidate_probabilities={"spot_logistic_online": raw_p},
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=exec_cfg,
        risk_cfg=risk_cfg,
        fee_bps_map={},
        platt_calibrator=calibrator,
        calibration_shift=0.09,
    )["spot_logistic_online"]

    second = pricing_mod.build_candidate_models_payload(
        candidate_probabilities={"spot_logistic_online": first},
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=exec_cfg,
        risk_cfg=risk_cfg,
        fee_bps_map={},
        platt_calibrator=calibrator,
        calibration_shift=0.09,
    )["spot_logistic_online"]

    assert second["p_up"] == first["p_up"]
    assert second["p_up_raw"] == first["p_up_raw"]
    assert second["p_up_pre_calibration"] == first["p_up_pre_calibration"]
    assert second["p_up_calibrated"] == first["p_up_calibrated"]
    assert second["decision_expected_edge"] == first["decision_expected_edge"]
    assert second["is_normalized"] is True


def test_candidate_payload_persistence_roundtrip_does_not_double_shift() -> None:
    calibrator = _LinearCalibrator()
    raw_p = 0.41
    books = _books()
    fee_cfg = FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0)

    payload = pricing_mod.build_candidate_models_payload(
        candidate_probabilities={"spot_logistic_online": raw_p},
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        fee_bps_map={},
        platt_calibrator=calibrator,
        calibration_shift=0.08,
    )
    stored = {"spot_logistic_online": payload["spot_logistic_online"]}
    reloaded = json.loads(json.dumps(stored))

    replay_payload = pricing_mod.build_candidate_models_payload(
        candidate_probabilities=reloaded,
        market_slug="m",
        books=books,
        seconds_to_expiry=240.0,
        fee_cfg=fee_cfg,
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        fee_bps_map={},
        platt_calibrator=calibrator,
        calibration_shift=0.08,
    )

    before = payload["spot_logistic_online"]
    after = replay_payload["spot_logistic_online"]
    assert after["p_up"] == before["p_up"]
    assert after["p_up_raw"] == before["p_up_raw"]
    assert after["p_up_pre_calibration"] == before["p_up_pre_calibration"]
    assert after["p_up_calibrated"] == before["p_up_calibrated"]
