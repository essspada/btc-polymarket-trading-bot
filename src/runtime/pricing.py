"""Model probability computation, candidate-models payload, and actionability.

This module owns the runtime "what does the model think p_up is" path: it
fans out to every enabled candidate predictor (proxy orderbook, transferred
markov, spot window-path, spot logistic, blends), applies Platt + bias
calibration, builds the per-candidate decision payload that monitor/paper
log, and implements the proxy/logistic meta policy override.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks
from src.runtime.utilities import maker_fill_probability_by_token as _maker_fill_probability_by_token, to_optional_float
from src.strategy.calibration import calibrate_probability, clip_prob
from src.strategy.model_wrapper import MarkovProxyModel, ModelContext
from src.strategy.signals import decide_trade
from src.strategy.spot_consensus import SpotConsensusConfig, predict_spot_consensus_blend_probability
from src.strategy.spot_logistic import predict_spot_logistic_probability
from src.strategy.spot_window_path import (
    SpotWindowPathConfig,
    blend_spot_with_market_probability,
    predict_spot_window_path_probability,
)
from src.strategy.transferred_markov import TransferredMarkovPredictor


def actionability_calibration_cfg(exec_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = exec_cfg.get("actionability_calibration")
    if not isinstance(payload, Mapping):
        return None
    enabled = bool(payload.get("enabled", False))
    if not enabled:
        return None
    cfg = dict(payload)
    artifact_path = cfg.get("artifact_path")
    if artifact_path is not None:
        cfg["artifact_path"] = str(artifact_path)
    return cfg


def skip_confidence_band(exec_cfg: Mapping[str, Any]) -> tuple[float, float] | None:
    payload = exec_cfg.get("confidence_skip_band")
    if not isinstance(payload, Mapping):
        return None
    if not bool(payload.get("enabled", False)):
        return None
    lo = to_optional_float(payload.get("low"))
    hi = to_optional_float(payload.get("high"))
    if lo is None or hi is None or lo >= hi:
        return None
    return (lo, hi)


def actionability_context(
    *,
    p_up: float,
    proxy_p_up: float | None,
    spot_recent_vol_5m_bps: float | None,
) -> dict[str, Any]:
    return {
        "p_up": float(p_up),
        "proxy_p_up": (float(proxy_p_up) if proxy_p_up is not None else None),
        "model_proxy_gap": (float(p_up - proxy_p_up) if proxy_p_up is not None else None),
        "spot_recent_vol_5m_bps": (float(spot_recent_vol_5m_bps) if spot_recent_vol_5m_bps is not None else None),
    }


def normalize_p_up(
    raw_p_up: float,
    *,
    platt_calibrator: Any | None,
    calibration_shift: float,
) -> dict[str, float]:
    p_pre_calibration = clip_prob(raw_p_up)
    p_calibrated = calibrate_probability(p_pre_calibration, platt_calibrator)
    p_up = clip_prob(p_calibrated + calibration_shift)
    return {
        "p_up_raw": float(raw_p_up),
        "p_up_pre_calibration": float(p_pre_calibration),
        "p_up_calibrated": float(p_calibrated),
        "p_up": float(p_up),
    }


def normalize_candidate_probability_input(
    candidate_prob: Any,
    *,
    platt_calibrator: Any | None,
    calibration_shift: float,
) -> dict[str, float] | None:
    if isinstance(candidate_prob, dict):
        is_normalized = bool(candidate_prob.get("is_normalized", False))
        has_norm_payload = (
            candidate_prob.get("p_up") is not None
            and candidate_prob.get("p_up_raw") is not None
            and candidate_prob.get("p_up_pre_calibration") is not None
            and candidate_prob.get("p_up_calibrated") is not None
        )
        if is_normalized or has_norm_payload:
            try:
                p_up = clip_prob(float(candidate_prob.get("p_up")))
            except Exception:
                return None
            try:
                p_up_raw = float(candidate_prob.get("p_up_raw", p_up))
            except Exception:
                p_up_raw = p_up
            try:
                p_up_pre_calibration = float(candidate_prob.get("p_up_pre_calibration", p_up))
            except Exception:
                p_up_pre_calibration = p_up
            try:
                p_up_calibrated = float(candidate_prob.get("p_up_calibrated", p_up))
            except Exception:
                p_up_calibrated = p_up
            return {
                "p_up_raw": float(p_up_raw),
                "p_up_pre_calibration": float(p_up_pre_calibration),
                "p_up_calibrated": float(p_up_calibrated),
                "p_up": float(p_up),
                "is_normalized": True,
            }
        try:
            raw_p = float(candidate_prob.get("p_up"))
        except Exception:
            return None
    else:
        try:
            raw_p = float(candidate_prob)
        except Exception:
            return None

    norm = normalize_p_up(
        raw_p,
        platt_calibrator=platt_calibrator,
        calibration_shift=calibration_shift,
    )
    norm["is_normalized"] = True
    return norm


def compute_model_probability(
    *,
    now: datetime,
    market_slug: str,
    market_id: str,
    books: MarketBooks,
    spot_ctx: Any | None,
    sec_to_exp: float,
    model_source: str,
    fallback_to_proxy_orderbook: bool,
    proxy_model: MarkovProxyModel,
    transfer_model: TransferredMarkovPredictor | None,
    spot_path_cfg: SpotWindowPathConfig,
    spot_logistic_model: Any | None,
    spot_consensus_cfg: SpotConsensusConfig,
    spot_logistic_market_blend_cfg: dict[str, Any] | None,
    proxy_logistic_market_blend_cfg: dict[str, Any] | None,
    model_weight: float,
    calibration_shift: float,
    platt_calibrator: Any | None,
    logger: Any,
) -> dict[str, Any] | None:
    ctx = ModelContext(
        up_mid=books.up.midpoint,
        down_mid=books.down.midpoint,
        up_imbalance=books.up.topk_imbalance,
        down_imbalance=books.down.topk_imbalance,
        seconds_to_expiry=sec_to_exp,
    )
    proxy_p_up = clip_prob(proxy_model.predict_proba(ctx))

    market_den = max(1e-9, books.up.midpoint + books.down.midpoint)
    market_p_up = clip_prob(books.up.midpoint / market_den)

    model_p_up: float | None = None
    spot_model_p_up: float | None = None
    spot_logistic_p_up: float | None = None
    spot_market_blend_p_up: float | None = None
    spot_logistic_market_blend_p_up: float | None = None
    spot_consensus_blend_p_up: float | None = None
    proxy_logistic_market_blend_p_up: float | None = None
    if transfer_model is not None:
        try:
            model_p_up = clip_prob(transfer_model.predict_up_probability(now=now))
        except Exception as exc:
            logger.info(
                "monitor_transfer_model_error",
                extra={
                    "extra": {
                        "market": market_slug,
                        "market_id": market_id,
                        "error": str(exc),
                        "fallback": "proxy_orderbook",
                    }
                },
            )
    if spot_ctx is not None:
        live_spot_row = {
            "spot_return_bps_from_open": getattr(spot_ctx, "spot_return_bps_from_open", None),
            "spot_recent_return_1m_bps": getattr(spot_ctx, "spot_recent_return_1m_bps", None),
            "spot_recent_vol_5m_bps": getattr(spot_ctx, "spot_recent_vol_5m_bps", None),
            "seconds_to_expiry": sec_to_exp,
            "market_p_up": market_p_up,
            "proxy_p_up": proxy_p_up,
        }
        try:
            spot_model_p_up = predict_spot_window_path_probability(
                spot_return_bps_from_open=live_spot_row["spot_return_bps_from_open"],
                spot_recent_return_1m_bps=live_spot_row["spot_recent_return_1m_bps"],
                spot_recent_vol_5m_bps=live_spot_row["spot_recent_vol_5m_bps"],
                seconds_to_expiry=sec_to_exp,
                cfg=spot_path_cfg,
            )
        except Exception as exc:
            logger.info(
                "monitor_spot_path_model_error",
                extra={
                    "extra": {
                        "market": market_slug,
                        "market_id": market_id,
                        "error": str(exc),
                        "fallback": "proxy_orderbook",
                    }
                },
            )
        try:
            spot_logistic_p_up = predict_spot_logistic_probability(spot_logistic_model, live_spot_row)
        except Exception as exc:
            logger.info(
                "monitor_spot_logistic_model_error",
                extra={
                    "extra": {
                        "market": market_slug,
                        "market_id": market_id,
                        "error": str(exc),
                        "fallback": "spot_market_blend",
                    }
                },
            )
        if spot_model_p_up is not None:
            spot_market_blend_p_up = blend_spot_with_market_probability(
                market_p_up=market_p_up,
                spot_p_up=spot_model_p_up,
                cfg=spot_path_cfg,
            )
        spot_logistic_blend_cfg = spot_logistic_market_blend_cfg or {}
        logistic_weight = float(spot_logistic_blend_cfg.get("logistic_weight", 0.2))
        if spot_logistic_p_up is not None:
            if market_p_up is not None:
                spot_logistic_market_blend_p_up = clip_prob(
                    logistic_weight * float(spot_logistic_p_up) + (1.0 - logistic_weight) * float(market_p_up)
                )
            else:
                spot_logistic_market_blend_p_up = clip_prob(float(spot_logistic_p_up))
        spot_consensus_blend_p_up = predict_spot_consensus_blend_probability(
            spot_window_path_p_up=spot_model_p_up,
            spot_logistic_p_up=spot_logistic_p_up,
            spot_market_blend_p_up=spot_market_blend_p_up,
            market_p_up=market_p_up,
            cfg=spot_consensus_cfg,
        )
        proxy_logistic_blend_cfg = proxy_logistic_market_blend_cfg or {}
        proxy_weight = max(0.0, float(proxy_logistic_blend_cfg.get("proxy_weight", 0.1)))
        proxy_logistic_weight = max(0.0, float(proxy_logistic_blend_cfg.get("logistic_weight", 0.2)))
        total_weight = proxy_weight + proxy_logistic_weight
        market_weight = max(0.0, 1.0 - total_weight)
        blend_total = proxy_weight + proxy_logistic_weight + market_weight
        if blend_total > 0.0:
            market_component = float(market_p_up)
            proxy_component = float(proxy_p_up)
            logistic_component = float(spot_logistic_p_up) if spot_logistic_p_up is not None else market_component
            proxy_logistic_market_blend_p_up = clip_prob(
                (
                    proxy_weight * proxy_component
                    + proxy_logistic_weight * logistic_component
                    + market_weight * market_component
                )
                / blend_total
            )

    if model_source == "proxy_orderbook":
        p_pre_cal = proxy_p_up
        model_used = "proxy_orderbook"
    elif model_source == "spot_window_path":
        if spot_model_p_up is None:
            if not fallback_to_proxy_orderbook:
                return None
            p_pre_cal = proxy_p_up
            model_used = "proxy_orderbook_fallback"
            model_p_up = proxy_p_up
        else:
            p_pre_cal = spot_model_p_up
            model_used = "spot_window_path"
            model_p_up = spot_model_p_up
    elif model_source == "spot_logistic_online":
        if spot_logistic_p_up is None:
            fallback_p = spot_model_p_up
            if fallback_p is None:
                if not fallback_to_proxy_orderbook:
                    return None
                p_pre_cal = proxy_p_up
                model_used = "proxy_orderbook_fallback"
                model_p_up = proxy_p_up
            else:
                p_pre_cal = blend_spot_with_market_probability(
                    market_p_up=market_p_up,
                    spot_p_up=fallback_p,
                    cfg=spot_path_cfg,
                )
                model_used = "spot_market_blend_fallback"
                model_p_up = fallback_p
        else:
            p_pre_cal = spot_logistic_p_up
            model_used = "spot_logistic_online"
            model_p_up = spot_logistic_p_up
    elif model_source == "spot_market_blend":
        if spot_model_p_up is None:
            if not fallback_to_proxy_orderbook:
                return None
            p_pre_cal = proxy_p_up
            model_used = "proxy_orderbook_fallback"
            model_p_up = proxy_p_up
        else:
            p_pre_cal = spot_market_blend_p_up if spot_market_blend_p_up is not None else market_p_up
            model_used = "spot_market_blend"
            model_p_up = spot_model_p_up
    elif model_source == "spot_logistic_market_blend":
        if spot_logistic_market_blend_p_up is None:
            if spot_logistic_p_up is None:
                if not fallback_to_proxy_orderbook:
                    return None
                p_pre_cal = proxy_p_up
                model_used = "proxy_orderbook_fallback"
                model_p_up = proxy_p_up
            else:
                p_pre_cal = spot_logistic_p_up
                model_used = "spot_logistic_online_fallback"
                model_p_up = spot_logistic_p_up
        else:
            p_pre_cal = spot_logistic_market_blend_p_up
            model_used = "spot_logistic_market_blend"
            model_p_up = spot_logistic_p_up
    elif model_source == "spot_consensus_blend":
        if spot_consensus_blend_p_up is None:
            fallback_p = spot_logistic_p_up if spot_logistic_p_up is not None else spot_market_blend_p_up
            if fallback_p is None:
                if not fallback_to_proxy_orderbook:
                    return None
                p_pre_cal = proxy_p_up
                model_used = "proxy_orderbook_fallback"
                model_p_up = proxy_p_up
            else:
                p_pre_cal = fallback_p
                model_used = "spot_consensus_blend_fallback"
                model_p_up = fallback_p
        else:
            p_pre_cal = spot_consensus_blend_p_up
            model_used = "spot_consensus_blend"
            model_p_up = spot_consensus_blend_p_up
    elif model_source == "proxy_logistic_market_blend":
        if proxy_logistic_market_blend_p_up is None:
            fallback_p = spot_logistic_market_blend_p_up
            if fallback_p is None:
                if not fallback_to_proxy_orderbook:
                    return None
                p_pre_cal = proxy_p_up
                model_used = "proxy_orderbook_fallback"
                model_p_up = proxy_p_up
            else:
                p_pre_cal = fallback_p
                model_used = "spot_logistic_market_blend_fallback"
                model_p_up = spot_logistic_p_up if spot_logistic_p_up is not None else proxy_p_up
        else:
            p_pre_cal = proxy_logistic_market_blend_p_up
            model_used = "proxy_logistic_market_blend"
            model_p_up = spot_logistic_p_up if spot_logistic_p_up is not None else proxy_p_up
    elif model_source == "transferred_markov":
        if model_p_up is None:
            if not fallback_to_proxy_orderbook:
                return None
            p_pre_cal = proxy_p_up
            model_used = "proxy_orderbook_fallback"
            model_p_up = proxy_p_up
        else:
            p_pre_cal = model_p_up
            model_used = "transferred_markov"
    else:
        seed_model = model_p_up if model_p_up is not None else proxy_p_up
        if model_p_up is None and not fallback_to_proxy_orderbook:
            return None
        p_pre_cal = clip_prob(model_weight * seed_model + (1.0 - model_weight) * market_p_up)
        model_used = "adaptive_blend"
        model_p_up = seed_model

    norm = normalize_p_up(
        p_pre_cal,
        platt_calibrator=platt_calibrator,
        calibration_shift=calibration_shift,
    )
    p_up = float(norm["p_up"])
    p_down = 1.0 - p_up
    predicted_side = "up" if p_up >= 0.5 else "down"
    confidence = p_up if predicted_side == "up" else p_down
    return {
        "p_up": p_up,
        "p_down": p_down,
        "predicted_side": predicted_side,
        "confidence": confidence,
        "p_pre_calibration": p_pre_cal,
        "model_used": model_used,
        "model_p_up": model_p_up,
        "market_p_up": market_p_up,
        "proxy_p_up": proxy_p_up,
        "candidate_probabilities": {
            "proxy_orderbook": proxy_p_up,
            "spot_window_path": spot_model_p_up,
            "spot_market_blend": spot_market_blend_p_up,
            "spot_logistic_online": spot_logistic_p_up,
            "spot_logistic_market_blend": spot_logistic_market_blend_p_up,
            "spot_consensus_blend": spot_consensus_blend_p_up,
            "proxy_logistic_market_blend": proxy_logistic_market_blend_p_up,
        },
    }


def build_candidate_models_payload(
    *,
    candidate_probabilities: dict[str, Any] | None,
    market_slug: str,
    books: MarketBooks,
    seconds_to_expiry: float,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    fee_bps_map: dict[str, float],
    platt_calibrator: Any | None,
    calibration_shift: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    allowed_order_types = exec_cfg.get("allowed_order_types", ["maker", "taker"])
    actionability_cfg = actionability_calibration_cfg(exec_cfg)
    proxy_norm = normalize_candidate_probability_input(
        (candidate_probabilities or {}).get("proxy_orderbook"),
        platt_calibrator=platt_calibrator,
        calibration_shift=calibration_shift,
    )
    proxy_reference_p_up = float(proxy_norm["p_up"]) if isinstance(proxy_norm, dict) and proxy_norm.get("p_up") is not None else None
    maker_fill_probability_by_token = _maker_fill_probability_by_token(
        books=books,
        seconds_to_expiry=seconds_to_expiry,
        exec_cfg=exec_cfg,
    )
    for model_name, raw_prob in (candidate_probabilities or {}).items():
        norm = normalize_candidate_probability_input(
            raw_prob,
            platt_calibrator=platt_calibrator,
            calibration_shift=calibration_shift,
        )
        if norm is None:
            continue
        p_up = float(norm["p_up"])
        predicted_side = "up" if p_up >= 0.5 else "down"
        confidence = p_up if predicted_side == "up" else (1.0 - p_up)
        item: dict[str, Any] = {
            "p_up_raw": norm["p_up_raw"],
            "p_up_pre_calibration": norm["p_up_pre_calibration"],
            "p_up_calibrated": norm["p_up_calibrated"],
            "p_up": p_up,
            "is_normalized": bool(norm.get("is_normalized", False)),
            "predicted_side": predicted_side,
            "confidence": confidence,
        }
        try:
            decision = decide_trade(
                market_slug=market_slug,
                books=books,
                p_up=p_up,
                min_edge_to_trade=float(exec_cfg.get("min_edge_to_trade", 0.004)),
                min_edge_for_taker=float(exec_cfg.get("min_edge_for_taker", 0.009)),
                max_exposure_usd=float(risk_cfg.get("max_exposure_per_window_usd", 0.0)),
                maker_preference=bool(exec_cfg.get("maker_preference", True)),
                fee_cfg=fee_cfg,
                taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12)),
                maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3)),
                taker_fee_bps_by_token=fee_bps_map,
                maker_fill_probability=float(exec_cfg.get("maker_fill_probability", 0.65)),
                maker_fill_probability_by_token=maker_fill_probability_by_token,
                maker_ev_advantage_required=float(exec_cfg.get("maker_ev_advantage_required", 0.0005)),
                allowed_order_types=list(allowed_order_types) if isinstance(allowed_order_types, list) else allowed_order_types,
                lock_side_to_prediction=bool(exec_cfg.get("lock_side_to_prediction", True)),
                min_reward_to_risk_ratio=float(exec_cfg.get("min_reward_to_risk_ratio", 0.0)),
                min_expected_roi_cash=to_optional_float(exec_cfg.get("min_expected_roi_cash")),
                min_breakeven_margin=to_optional_float(exec_cfg.get("min_breakeven_margin")),
                sizing_mode=str(exec_cfg.get("sizing_mode", "edge_scaled")),
                max_entry_price=to_optional_float(exec_cfg.get("max_entry_price")),
                actionability_calibration=actionability_cfg,
                actionability_context=actionability_context(
                    p_up=float(p_up),
                    proxy_p_up=proxy_reference_p_up,
                    spot_recent_vol_5m_bps=None,
                ),
                skip_confidence_band=skip_confidence_band(exec_cfg),
            )
            item.update(
                {
                    "decision_action": decision.action,
                    "decision_reason": decision.reason,
                    "decision_best_edge": decision.best_edge,
                    "decision_score_mode": decision.score_mode,
                    "decision_score_value": decision.score_value,
                    "decision_expected_roi_cash": decision.expected_roi_cash,
                    "decision_breakeven_probability": decision.breakeven_probability,
                    "decision_breakeven_margin": decision.breakeven_margin,
                    "decision_fill_probability": decision.fill_probability,
                    "decision_ev_executable": decision.ev_executable,
                    "decision_ev_fill": decision.ev_fill,
                    "decision_cash_required": decision.cash_required,
                    "decision_calibrated_p_side": decision.calibrated_p_side,
                    "decision_calibrated_net_edge": decision.calibrated_net_edge,
                    "decision_calibrated_expected_roi_cash": decision.calibrated_expected_roi_cash,
                    "decision_calibrated_breakeven_margin": decision.calibrated_breakeven_margin,
                    "decision_actionability_calibration_applied": decision.actionability_calibration_applied,
                    "decision_actionability_calibration_rejected": decision.actionability_calibration_rejected,
                    "decision_actionability_calibration_reason": decision.actionability_calibration_reason,
                    "decision_actionability_lower_bound_floor": decision.actionability_lower_bound_floor,
                    "decision_actionability_lower_bound_sources": decision.actionability_lower_bound_sources,
                    "decision_order_type": (
                        decision.intent.order_type if decision.intent is not None else None
                    ),
                    "decision_expected_edge": (
                        decision.intent.expected_edge if decision.intent is not None else None
                    ),
                    "decision_token_id": (
                        decision.intent.token_id if decision.intent is not None else None
                    ),
                }
            )
        except Exception as exc:
            item["decision_error"] = str(exc)
        payload[str(model_name)] = item
    return payload


def candidate_selected_side(item: dict[str, Any], books: MarketBooks) -> str | None:
    token_id = str(item.get("decision_token_id") or "").strip()
    if token_id == books.up.token_id:
        return "up"
    if token_id == books.down.token_id:
        return "down"
    return None


def build_proxy_logistic_meta_candidate(
    *,
    candidate_models_payload: dict[str, Any] | None,
    books: MarketBooks,
    meta_cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    payload = candidate_models_payload or {}
    proxy = payload.get("proxy_orderbook")
    logistic = payload.get("spot_logistic_online")
    fallback_name = str((meta_cfg or {}).get("fallback_source", "spot_logistic_online")).strip().lower()
    fallback_candidates = {
        "spot_logistic_online": logistic,
        "spot_logistic_market_blend": payload.get("spot_logistic_market_blend"),
        "proxy_logistic_market_blend": payload.get("proxy_logistic_market_blend"),
        "proxy_orderbook": proxy,
    }
    fallback = fallback_candidates.get(fallback_name)
    if not isinstance(fallback, dict):
        fallback = logistic or payload.get("spot_logistic_market_blend") or payload.get("proxy_logistic_market_blend") or proxy
    if not isinstance(proxy, dict) or not isinstance(logistic, dict):
        if not isinstance(fallback, dict):
            return None
        out = dict(fallback)
        out["meta_source"] = "fallback"
        out["meta_reason"] = "missing_proxy_or_logistic"
        return out

    proxy_midpoint_band = float((meta_cfg or {}).get("proxy_midpoint_band", 0.06))
    proxy_edge_margin = float((meta_cfg or {}).get("proxy_edge_margin", 0.0))
    proxy_selected = candidate_selected_side(proxy, books)
    logistic_selected = candidate_selected_side(logistic, books)
    proxy_predicted = str(proxy.get("predicted_side") or "").strip().lower()
    proxy_trade = str(proxy.get("decision_action") or "").strip().lower() == "trade"
    logistic_trade = str(logistic.get("decision_action") or "").strip().lower() == "trade"
    proxy_edge = float(proxy.get("decision_expected_edge") or 0.0)
    logistic_edge = float(logistic.get("decision_expected_edge") or 0.0)
    fallback_edge = float(fallback.get("decision_expected_edge") or logistic_edge or 0.0) if isinstance(fallback, dict) else logistic_edge

    chosen: dict[str, Any]
    source: str
    reason: str
    if (
        proxy_trade
        and logistic_trade
        and proxy_selected in {"up", "down"}
        and proxy_selected == logistic_selected
    ):
        chosen = logistic
        source = "spot_logistic_online"
        reason = "proxy_logistic_agree"
    elif (
        proxy_trade
        and proxy_selected in {"up", "down"}
        and proxy_selected != proxy_predicted
        and (not logistic_trade or proxy_selected != logistic_selected)
        and proxy_edge >= (logistic_edge + proxy_edge_margin)
    ):
        chosen = proxy
        source = "proxy_orderbook"
        reason = "proxy_ev_flip"
    elif (
        proxy_trade
        and abs(float(proxy.get("p_up", 0.5)) - 0.5) <= proxy_midpoint_band
        and proxy_edge >= (fallback_edge + proxy_edge_margin)
    ):
        chosen = proxy
        source = "proxy_orderbook"
        reason = "proxy_near_midpoint"
    else:
        if not isinstance(fallback, dict):
            return None
        chosen = fallback
        source = "fallback"
        reason = "fallback_candidate"

    out = dict(chosen)
    out["meta_source"] = source
    out["meta_reason"] = reason
    return out


def apply_meta_policy_override(
    *,
    probs: dict[str, Any],
    candidate_models_payload: dict[str, Any] | None,
    books: MarketBooks,
    model_source: str,
    meta_cfg: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(candidate_models_payload or {})
    meta_item = build_proxy_logistic_meta_candidate(
        candidate_models_payload=payload,
        books=books,
        meta_cfg=meta_cfg,
    )
    if meta_item is not None:
        payload["proxy_logistic_meta_policy"] = meta_item
    if model_source != "proxy_logistic_meta_policy" or meta_item is None:
        return probs, payload

    p_up = clip_prob(float(meta_item.get("p_up", probs.get("p_up", 0.5))))
    p_up_raw = float(meta_item.get("p_up_raw", p_up))
    p_up_pre_calibration = float(meta_item.get("p_up_pre_calibration", p_up))
    predicted_side = "up" if p_up >= 0.5 else "down"
    updated = dict(probs)
    updated["p_up"] = p_up
    updated["p_down"] = 1.0 - p_up
    updated["predicted_side"] = predicted_side
    updated["confidence"] = p_up if predicted_side == "up" else (1.0 - p_up)
    updated["p_pre_calibration"] = p_up_pre_calibration
    updated["model_used"] = "proxy_logistic_meta_policy"
    updated["model_p_up"] = p_up_raw
    candidate_probabilities = dict(updated.get("candidate_probabilities") or {})
    candidate_probabilities["proxy_logistic_meta_policy"] = {
        "p_up_raw": p_up_raw,
        "p_up_pre_calibration": p_up_pre_calibration,
        "p_up_calibrated": float(meta_item.get("p_up_calibrated", p_up)),
        "p_up": p_up,
        "is_normalized": True,
    }
    updated["candidate_probabilities"] = candidate_probabilities
    return updated, payload
