"""Continuous 5-minute accuracy monitor (no paper or live trading).

Polls Polymarket, builds a fresh prediction at the configured timing policy
stage, and resolves outcomes when each window settles. This is the data
source for downstream calibration / replay tooling; it never sends orders.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.oracles import OracleClient
from src.data.spot_context import SpotContextClient
from src.execution.shadow_auditor import ShadowAuditor
from src.polymarket.book_quality import BookQualityConfig, assess_market_books
from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.clients.gamma_client import GammaClient
from src.polymarket.fees import FeeModelConfig
from src.polymarket.market_discovery import (
    discover_btc_5m_window_markets,
    extract_resolved_outcome_side,
)
from src.runtime.persistence import (
    append_jsonl,
    append_monitor_observation,
    append_timing_policy_event,
    read_jsonl,
    write_json,
)
from src.runtime.pricing import (
    actionability_calibration_cfg,
    actionability_context,
    apply_meta_policy_override,
    build_candidate_models_payload,
    compute_model_probability,
    skip_confidence_band,
)
from src.runtime.shadow_risk import (
    build_monitor_adaptive_risk_shadow_payload,
)
from src.runtime.snapshots import (
    book_depth_payload,
    build_market_books_snapshot,
)
from src.runtime.state import (
    OUTPUT_SCHEMA_VERSION,
    PendingPrediction,
    load_deferred_timing,
    load_pending_predictions,
    load_running_stats,
)
from src.runtime.timing import (
    apply_runtime_timing_policy,
    prune_deferred_timing,
)
from src.runtime.utilities import (
    build_monitor_summary,
    fetch_taker_fee_bps_by_token,
    gamma_error_payload,
    maker_fill_probability_by_token,
    parse_iso_utc,
    resolve_output_path,
    to_optional_float,
)
from src.runtime.x3 import (
    outcome_side_for_intent,
)
from src.strategy.calibration import (
    compute_adaptive_model_weight,
    compute_online_bias_shift,
    fit_rolling_platt,
)
from src.strategy.confirmation_gate import apply_confirmation_gate
from src.strategy.model_wrapper import MarkovProxyModel
from src.strategy.signals import decide_trade
from src.strategy.spot_consensus import SpotConsensusConfig
from src.strategy.spot_logistic import SpotLogisticConfig, fit_rolling_spot_logistic
from src.strategy.spot_window_path import (
    SpotWindowPathConfig,
)
from src.strategy.timing_policy import (
    DeferredTimingDecision,
    TimingPolicyStage,
    next_future_stage_index,
    parse_timing_policy_base_sources,
    parse_timing_policy_configs,
    timing_policy_entry_mode,
)
from src.strategy.transferred_markov import TransferredMarkovConfig, TransferredMarkovPredictor
from src.utils.logging import build_logger


def run_monitor_5m(cfg: dict[str, Any], max_runtime_minutes: float | None = None) -> dict[str, Any]:
    logger = build_logger(f"{cfg['app']['name']}_monitor", cfg["paths"]["logs_dir"])
    gamma = GammaClient(cfg["polymarket"]["gamma_base_url"])
    clob = ClobClient(cfg["polymarket"]["clob_base_url"])
    proxy_model = MarkovProxyModel(default_confidence=float(cfg["model"].get("default_confidence", 0.5)))

    model_cfg = cfg.get("model", {})
    model_source = str(model_cfg.get("source", "adaptive_blend")).strip().lower()
    fallback_to_proxy_orderbook = bool(model_cfg.get("fallback_to_proxy_orderbook", True))
    timing_policy_cfgs = parse_timing_policy_configs(model_cfg)
    timing_policy_base_sources = parse_timing_policy_base_sources(model_cfg)
    timing_policy_stages = timing_policy_cfgs.get(model_source)
    timing_policy_base_source = timing_policy_base_sources.get(model_source, "spot_consensus_blend")

    transfer_cfg_raw = model_cfg.get("transferred_markov", {}) if isinstance(model_cfg, dict) else {}
    spot_path_cfg = SpotWindowPathConfig.from_dict(model_cfg.get("spot_window_path", {}) if isinstance(model_cfg, dict) else {})
    spot_log_cfg = SpotLogisticConfig.from_dict(model_cfg.get("spot_logistic", {}) if isinstance(model_cfg, dict) else {})
    spot_consensus_cfg = SpotConsensusConfig.from_dict(model_cfg.get("spot_consensus_blend", {}) if isinstance(model_cfg, dict) else {})
    spot_logistic_market_blend_cfg = model_cfg.get("spot_logistic_market_blend", {}) if isinstance(model_cfg, dict) else {}
    proxy_logistic_market_blend_cfg = model_cfg.get("proxy_logistic_market_blend", {}) if isinstance(model_cfg, dict) else {}
    proxy_logistic_meta_policy_cfg = model_cfg.get("proxy_logistic_meta_policy", {}) if isinstance(model_cfg, dict) else {}
    spot_log_seed_rows: list[dict[str, Any]] = []
    spot_log_seed_raw = model_cfg.get("spot_logistic", {}).get("seed_rows_file") if isinstance(model_cfg, dict) else None
    if spot_log_seed_raw:
        spot_log_seed_rows = read_jsonl(Path(str(spot_log_seed_raw)))
    use_transfer_model = bool(transfer_cfg_raw.get("enabled", True)) and model_source in {
        "transferred_markov",
        "adaptive_blend",
    }
    transfer_model: TransferredMarkovPredictor | None = None
    if use_transfer_model:
        transfer_cfg = TransferredMarkovConfig(
            python_bin=str(
                transfer_cfg_raw.get(
                    "python_bin",
                    "",
                )
            ),
            project_root=str(
                transfer_cfg_raw.get(
                    "project_root",
                    "",
                )
            ),
            predictor_module=str(transfer_cfg_raw.get("predictor_module", "src.models.predict_sequence")),
            model_config_path=str(
                transfer_cfg_raw.get(
                    "model_config_path",
                    "",
                )
            ),
            model_weights_path=str(
                transfer_cfg_raw.get(
                    "model_weights_path",
                    "",
                )
            ),
            output_dir=str(
                transfer_cfg_raw.get(
                    "output_dir",
                    "outputs/markov_bridge",
                )
            ),
            symbol=str(transfer_cfg_raw.get("symbol", "BTCUSDT")),
            interval=str(transfer_cfg_raw.get("interval", "5m")),
            candles_limit=int(transfer_cfg_raw.get("candles_limit", 14000)),
            request_limit=int(transfer_cfg_raw.get("request_limit", 1000)),
            request_timeout_seconds=int(transfer_cfg_raw.get("request_timeout_seconds", 20)),
            request_pause_seconds=float(transfer_cfg_raw.get("request_pause_seconds", 0.1)),
            predict_timeout_seconds=int(transfer_cfg_raw.get("predict_timeout_seconds", 120)),
            hold_weight=float(transfer_cfg_raw.get("hold_weight", 0.0)),
            max_prediction_age_minutes=int(transfer_cfg_raw.get("max_prediction_age_minutes", 45)),
            min_refresh_seconds=int(transfer_cfg_raw.get("min_refresh_seconds", 30)),
        )
        transfer_model = TransferredMarkovPredictor(transfer_cfg)

    blend_cfg = model_cfg.get("adaptive_blend", {}) if isinstance(model_cfg, dict) else {}
    blend_lookback = max(20, int(blend_cfg.get("lookback_resolved", 120)))
    blend_min_samples = max(5, int(blend_cfg.get("min_samples", 20)))
    default_model_weight = float(blend_cfg.get("default_model_weight", 0.35))
    min_model_weight = float(blend_cfg.get("min_model_weight", 0.1))
    max_model_weight = float(blend_cfg.get("max_model_weight", 0.9))
    calibration_strength = float(blend_cfg.get("calibration_strength", 0.7))
    max_calibration_shift = float(blend_cfg.get("max_calibration_shift", 0.15))
    platt_lookback = max(20, int(blend_cfg.get("platt_lookback", blend_lookback)))
    platt_min_samples = max(10, int(blend_cfg.get("platt_min_samples", 40)))
    platt_min_class_samples = max(2, int(blend_cfg.get("platt_min_class_samples", 8)))
    book_quality_cfg = BookQualityConfig.from_dict(cfg.get("book_quality", {}))
    oracle_cfg = cfg.get("oracle", {})
    spot_cfg = cfg.get("spot_context", {})
    oracle_client: OracleClient | None = None
    if bool(oracle_cfg.get("enabled", False)):
        oracle_client = OracleClient(
            symbol=str(oracle_cfg.get("symbol", "BTCUSDT")),
            oracle_price_url=oracle_cfg.get("oracle_price_url"),
            timeout=int(oracle_cfg.get("timeout_seconds", 10)),
        )
    spot_client: SpotContextClient | None = None
    if bool(spot_cfg.get("enabled", True)):
        spot_client = SpotContextClient(
            symbol=str(spot_cfg.get("symbol", "BTCUSDT")),
            timeout=int(spot_cfg.get("timeout_seconds", 10)),
            kline_limit=int(spot_cfg.get("kline_limit", 8)),
            cache_ttl_seconds=float(spot_cfg.get("cache_ttl_seconds", 5.0)),
        )

    mon_cfg = cfg.get("monitor", {})
    paper_cfg = cfg.get("paper", {})
    disc_cfg = cfg.get("discovery", {})
    exec_cfg = cfg.get("execution", {})
    risk_cfg = cfg.get("risk", {})
    fee_cfg = FeeModelConfig(**cfg["fees"])
    maker_fill_probability = float(exec_cfg.get("maker_fill_probability", 0.65))
    maker_ev_advantage_required = float(exec_cfg.get("maker_ev_advantage_required", 0.0005))
    allowed_order_types = exec_cfg.get("allowed_order_types", ["maker", "taker"])
    oracle_external_feed = bool(oracle_cfg.get("oracle_price_url"))
    oracle_source = (
        ("external" if oracle_external_feed else "spot_fallback")
        if oracle_client is not None
        else "disabled"
    )
    poll_seconds = max(1, int(mon_cfg.get("poll_seconds", 10)))
    timing_runtime_cfg = model_cfg.get("timing_policy_runtime", {}) if isinstance(model_cfg, dict) else {}
    allow_late_fresh_start = bool(timing_runtime_cfg.get("allow_late_fresh_start", False))
    late_start_grace_seconds = max(0.0, float(timing_runtime_cfg.get("late_start_grace_seconds", poll_seconds)))
    settlement_grace_seconds = max(0, int(mon_cfg.get("settlement_grace_seconds", 45)))
    lookahead_windows = max(1, int(mon_cfg.get("lookahead_windows", 12)))
    lookback_windows = max(0, int(mon_cfg.get("lookback_windows", 1)))
    min_start_delay_seconds = max(0.0, float(mon_cfg.get("min_start_delay_seconds", 0.0)))
    max_start_delay_seconds = max(min_start_delay_seconds, float(mon_cfg.get("max_start_delay_seconds", 75)))
    resolution_winner_threshold = float(mon_cfg.get("resolution_winner_threshold", 0.99))

    runtime_limit = max_runtime_minutes
    if runtime_limit is None:
        runtime_limit = float(mon_cfg.get("max_runtime_minutes", 0))
    runtime_limit = max(0.0, float(runtime_limit))

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    predictions_path = resolve_output_path(outputs_dir, str(mon_cfg.get("predictions_file", "predictions_5m.jsonl")))
    outcomes_path = resolve_output_path(outputs_dir, str(mon_cfg.get("outcomes_file", "outcomes_5m.jsonl")))
    state_path = resolve_output_path(outputs_dir, str(mon_cfg.get("state_file", "monitor_5m_state.json")))
    timing_events_path = resolve_output_path(
        outputs_dir,
        str(mon_cfg.get("timing_events_file", "monitor_timing_events_5m.jsonl")),
    )
    adaptive_risk_shadow_path = resolve_output_path(
        outputs_dir,
        str(mon_cfg.get("adaptive_risk_shadow_file", "monitor_adaptive_risk_shadow_decisions_5m.jsonl")),
    )
    observations_path = (
        resolve_output_path(outputs_dir, str(mon_cfg.get("observations_file")))
        if mon_cfg.get("observations_file")
        else None
    )
    observations_raw_l2_depth = max(0, int(mon_cfg.get("observations_raw_l2_depth", 25 if observations_path else 0)))

    pending = load_pending_predictions(state_path)
    shadow_auditor = ShadowAuditor()
    deferred_timing = load_deferred_timing(state_path)
    resolved_ids, total, correct = load_running_stats(outcomes_path)
    for market_id in list(pending.keys()):
        if market_id in resolved_ids:
            pending.pop(market_id, None)
    prune_deferred_timing(
        deferred=deferred_timing,
        resolved_ids=resolved_ids,
        now=datetime.now(UTC),
        settlement_grace_seconds=settlement_grace_seconds,
    )

    _ROLLING_WR_LOOKBACKS = (10, 15, 20, 30)
    _max_rolling = max(_ROLLING_WR_LOOKBACKS)
    _rolling_outcomes: deque[bool] = deque(maxlen=_max_rolling)
    if outcomes_path.exists():
        try:
            with outcomes_path.open("r", encoding="utf-8") as _fh:
                for _line in _fh:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _row = json.loads(_line)
                    if "is_correct" in _row:
                        _rolling_outcomes.append(bool(_row["is_correct"]))
        except Exception:
            pass

    started_at = datetime.now(UTC)
    summary: dict[str, Any] = {}

    logger.info(
        "monitor_started",
        extra={
            "extra": {
                "poll_seconds": poll_seconds,
                "lookahead_windows": lookahead_windows,
                "lookback_windows": lookback_windows,
                "min_start_delay_seconds": min_start_delay_seconds,
                "max_start_delay_seconds": max_start_delay_seconds,
                "settlement_grace_seconds": settlement_grace_seconds,
                "runtime_limit_minutes": runtime_limit,
                "model_source": model_source,
                "timing_policy_active": bool(timing_policy_stages),
                "observations_file": str(observations_path) if observations_path is not None else None,
                "observations_raw_l2_depth": observations_raw_l2_depth,
                "allow_late_fresh_start": allow_late_fresh_start,
                "late_start_grace_seconds": late_start_grace_seconds,
                "existing_resolved": len(resolved_ids),
                "existing_pending": len(pending),
                "existing_deferred_timing": len(deferred_timing),
            }
        },
    )

    try:
        while True:
            now = datetime.now(UTC)
            elapsed_minutes = (now - started_at).total_seconds() / 60.0
            if runtime_limit > 0 and elapsed_minutes >= runtime_limit:
                break
            prune_deferred_timing(
                deferred=deferred_timing,
                resolved_ids=resolved_ids,
                now=now,
                settlement_grace_seconds=settlement_grace_seconds,
            )

            recent_outcomes = read_jsonl(outcomes_path, max_rows=max(200, blend_lookback * 2))
            model_weight = compute_adaptive_model_weight(
                recent_outcomes,
                model_prob_key="model_p_up",
                market_prob_key="market_p_up",
                lookback=blend_lookback,
                min_samples=blend_min_samples,
                default_weight=default_model_weight,
                min_weight=min_model_weight,
                max_weight=max_model_weight,
            )
            calibration_shift = compute_online_bias_shift(
                recent_outcomes,
                prob_key="p_up",
                lookback=blend_lookback,
                min_samples=blend_min_samples,
                strength=calibration_strength,
                max_abs_shift=max_calibration_shift,
            )
            platt_calibrator = fit_rolling_platt(
                recent_outcomes,
                prob_key="p_up",
                lookback=platt_lookback,
                min_samples=platt_min_samples,
                min_class_samples=platt_min_class_samples,
            )
            spot_logistic_model = fit_rolling_spot_logistic([*spot_log_seed_rows, *recent_outcomes], spot_log_cfg)
            oracle_basis = oracle_client.fetch_basis() if oracle_client is not None else None

            discovered = discover_btc_5m_window_markets(
                gamma=gamma,
                now=now,
                lookback_windows=lookback_windows,
                lookahead_windows=lookahead_windows,
                require_accepting_orders=bool(disc_cfg.get("require_accepting_orders", True)),
                include_closed=True,
                require_chainlink=True,
                max_results=max(lookahead_windows + lookback_windows + 8, 20),
            )

            logger.info(
                "monitor_discovered_markets",
                extra={"extra": {"count": len(discovered), "pending": len(pending), "resolved": len(resolved_ids)}},
            )
            if not discovered and gamma.last_error:
                logger.warning(
                    "monitor_discovery_network_error",
                    extra={
                        "extra": {
                            "pending": len(pending),
                            "resolved": len(resolved_ids),
                            **gamma_error_payload(gamma),
                        }
                    },
                )

            for m in discovered:
                if m.market_id in resolved_ids:
                    continue
                if now < m.start_time or now >= m.end_time:
                    continue

                is_pending_market = m.market_id in pending
                seconds_from_start = (now - m.start_time).total_seconds()
                deferred_item = deferred_timing.get(m.market_id)
                current_stage_index = deferred_item.next_stage_index if deferred_item is not None else None
                if not timing_policy_stages:
                    if seconds_from_start < min_start_delay_seconds:
                        continue
                    if seconds_from_start > max_start_delay_seconds:
                        continue

                sec_to_exp = max(0.0, (m.end_time - now).total_seconds())
                spot_ctx = spot_client.fetch_window_context(now=now, window_start=m.start_time) if spot_client is not None else None
                books, raw_orderbook, book_error = build_market_books_snapshot(
                    clob,
                    m,
                    raw_l2_depth=observations_raw_l2_depth if observations_path is not None else 0,
                )
                if books is None:
                    logger.info(
                        "monitor_skip_no_book",
                        extra={
                            "extra": {
                                "market": m.market_slug,
                                "market_id": m.market_id,
                                "error": book_error,
                            }
                        },
                    )
                    append_monitor_observation(
                        observations_path,
                        now=now,
                        market=m,
                        seconds_from_start=seconds_from_start,
                        seconds_to_expiry=sec_to_exp,
                        model_source=model_source,
                        timing_policy_active=bool(timing_policy_stages),
                        timing_action="no_book",
                        timing_selection=None,
                        current_stage_index=current_stage_index,
                        books=None,
                        raw_orderbook=None,
                        quality=None,
                        fee_bps_map={},
                        spot_ctx=spot_ctx,
                        oracle_basis=oracle_basis,
                        oracle_source=oracle_source,
                        oracle_external_feed=oracle_external_feed,
                        model_weight=model_weight,
                        calibration_shift=calibration_shift,
                        probs=None,
                        candidate_models_payload=None,
                        book_error=book_error,
                    )
                    continue
                shadow_auditor.on_orderbook_update(m.market_id, books, now=now)

                quality = assess_market_books(books=books, cfg=book_quality_cfg)
                if not quality.ok:
                    logger.info(
                        "monitor_skip_book_quality",
                        extra={
                            "extra": {
                                "market": m.market_slug,
                                "market_id": m.market_id,
                                "reason": quality.reason,
                                "flags": quality.flags,
                                "score": quality.score,
                            }
                        },
                    )
                    append_monitor_observation(
                        observations_path,
                        now=now,
                        market=m,
                        seconds_from_start=seconds_from_start,
                        seconds_to_expiry=sec_to_exp,
                        model_source=model_source,
                        timing_policy_active=bool(timing_policy_stages),
                        timing_action="book_quality_skip",
                        timing_selection=None,
                        current_stage_index=current_stage_index,
                        books=books,
                        raw_orderbook=raw_orderbook,
                        quality=quality,
                        fee_bps_map={},
                        spot_ctx=spot_ctx,
                        oracle_basis=oracle_basis,
                        oracle_source=oracle_source,
                        oracle_external_feed=oracle_external_feed,
                        model_weight=model_weight,
                        calibration_shift=calibration_shift,
                        probs=None,
                        candidate_models_payload=None,
                    )
                    continue

                fee_bps_map = fetch_taker_fee_bps_by_token(clob=clob, books=books)
                probs = compute_model_probability(
                    now=now,
                    market_slug=m.market_slug,
                    market_id=m.market_id,
                    books=books,
                    spot_ctx=spot_ctx,
                    sec_to_exp=sec_to_exp,
                    model_source=(timing_policy_base_source if timing_policy_stages else model_source),
                    fallback_to_proxy_orderbook=fallback_to_proxy_orderbook,
                    proxy_model=proxy_model,
                    transfer_model=transfer_model,
                    spot_path_cfg=spot_path_cfg,
                    spot_logistic_model=spot_logistic_model,
                    spot_consensus_cfg=spot_consensus_cfg,
                    spot_logistic_market_blend_cfg=spot_logistic_market_blend_cfg,
                    proxy_logistic_market_blend_cfg=proxy_logistic_market_blend_cfg,
                    model_weight=model_weight,
                    calibration_shift=calibration_shift,
                    platt_calibrator=platt_calibrator,
                    logger=logger,
                )
                if probs is None:
                    logger.info(
                        "monitor_skip_no_model_prediction",
                        extra={
                            "extra": {
                                "market": m.market_slug,
                                "market_id": m.market_id,
                                "reason": "transfer_failed_and_fallback_disabled",
                            }
                        },
                    )
                    append_monitor_observation(
                        observations_path,
                        now=now,
                        market=m,
                        seconds_from_start=seconds_from_start,
                        seconds_to_expiry=sec_to_exp,
                        model_source=model_source,
                        timing_policy_active=bool(timing_policy_stages),
                        timing_action="no_model_prediction",
                        timing_selection=None,
                        current_stage_index=current_stage_index,
                        books=books,
                        raw_orderbook=raw_orderbook,
                        quality=quality,
                        fee_bps_map=fee_bps_map,
                        spot_ctx=spot_ctx,
                        oracle_basis=oracle_basis,
                        oracle_source=oracle_source,
                        oracle_external_feed=oracle_external_feed,
                        model_weight=model_weight,
                        calibration_shift=calibration_shift,
                        probs=None,
                        candidate_models_payload=None,
                        model_error="transfer_failed_and_fallback_disabled",
                    )
                    continue

                candidate_models_payload = build_candidate_models_payload(
                    candidate_probabilities=probs.get("candidate_probabilities"),
                    market_slug=m.market_slug,
                    books=books,
                    seconds_to_expiry=sec_to_exp,
                    fee_cfg=fee_cfg,
                    exec_cfg=exec_cfg,
                    risk_cfg=risk_cfg,
                    fee_bps_map=fee_bps_map,
                    platt_calibrator=platt_calibrator,
                    calibration_shift=calibration_shift,
                )
                probs, candidate_models_payload = apply_meta_policy_override(
                    probs=probs,
                    candidate_models_payload=candidate_models_payload,
                    books=books,
                    model_source=model_source,
                    meta_cfg=proxy_logistic_meta_policy_cfg,
                )
                timing_selection: dict[str, Any] | None = None
                timing_action = "finalize"
                if timing_policy_stages:
                    timing_action, timing_selection, probs, candidate_models_payload = apply_runtime_timing_policy(
                        model_source=model_source,
                        timing_policy_stages=timing_policy_stages,
                        seconds_from_start=seconds_from_start,
                        min_edge_to_trade=float(exec_cfg.get("min_edge_to_trade", 0.004)),
                        candidate_models_payload=candidate_models_payload,
                        probs=probs,
                        current_stage_index=current_stage_index,
                        late_start_grace_seconds=late_start_grace_seconds,
                    )
                append_monitor_observation(
                    observations_path,
                    now=now,
                    market=m,
                    seconds_from_start=seconds_from_start,
                    seconds_to_expiry=sec_to_exp,
                    model_source=model_source,
                    timing_policy_active=bool(timing_policy_stages),
                    timing_action=timing_action,
                    timing_selection=timing_selection,
                    current_stage_index=current_stage_index,
                    books=books,
                    raw_orderbook=raw_orderbook,
                    quality=quality,
                    fee_bps_map=fee_bps_map,
                    spot_ctx=spot_ctx,
                    oracle_basis=oracle_basis,
                    oracle_source=oracle_source,
                    oracle_external_feed=oracle_external_feed,
                    model_weight=model_weight,
                    calibration_shift=calibration_shift,
                    probs=probs,
                    candidate_models_payload=candidate_models_payload,
                )
                if is_pending_market:
                    continue
                if timing_policy_stages:
                    if timing_action == "finalize" and isinstance(timing_selection, dict):
                        stage = timing_selection.get("stage")
                        stage_index_raw = timing_selection.get("stage_index")
                        stage_index = int(stage_index_raw) if stage_index_raw is not None else 0
                        entry_mode = timing_policy_entry_mode(
                            stage=stage if isinstance(stage, TimingPolicyStage) else None,
                            current_stage_index=current_stage_index,
                            seconds_from_start=seconds_from_start,
                            late_start_grace_seconds=late_start_grace_seconds,
                        )
                        if entry_mode == "late_fresh_start" and not allow_late_fresh_start:
                            future_stage_index = next_future_stage_index(
                                timing_policy_stages,
                                seconds_from_start=seconds_from_start,
                                current_stage_index=stage_index,
                            )
                            if future_stage_index is not None:
                                deferred_timing[m.market_id] = DeferredTimingDecision(
                                    market_id=m.market_id,
                                    market_slug=m.market_slug,
                                    event_slug=m.event_slug,
                                    series_slug=str(m.series_slug or ""),
                                    start_time=m.start_time.isoformat(),
                                    end_time=m.end_time.isoformat(),
                                    timing_policy_name=model_source,
                                    next_stage_index=int(future_stage_index),
                                    deferred_at=now.isoformat(),
                                )
                                timing_selection = dict(timing_selection)
                                timing_selection["next_stage_index"] = int(future_stage_index)
                                timing_selection["reason"] = "late_fresh_start_wait_next"
                                logger.info(
                                    "monitor_timing_policy_defer",
                                    extra={
                                        "extra": {
                                            "market": m.market_slug,
                                            "market_id": m.market_id,
                                            "timing_policy_name": model_source,
                                            "next_stage_index": int(future_stage_index),
                                            "reason": "late_fresh_start_wait_next",
                                        }
                                    },
                                )
                                append_timing_policy_event(
                                    timing_events_path,
                                    mode="monitor-5m",
                                    now=now,
                                    market=m,
                                    seconds_from_start=seconds_from_start,
                                    selection=timing_selection,
                                    action="defer",
                                    current_stage_index=current_stage_index,
                                    late_start_grace_seconds=late_start_grace_seconds,
                                    probs=probs,
                                )
                                continue
                            timing_selection = dict(timing_selection)
                            timing_selection["reason"] = "late_fresh_start_disabled"
                            logger.info(
                                "monitor_timing_policy_skip",
                                extra={
                                    "extra": {
                                        "market": m.market_slug,
                                        "market_id": m.market_id,
                                        "timing_policy_name": model_source,
                                        "reason": "late_fresh_start_disabled",
                                    }
                                },
                            )
                            append_timing_policy_event(
                                timing_events_path,
                                mode="monitor-5m",
                                now=now,
                                market=m,
                                seconds_from_start=seconds_from_start,
                                selection=timing_selection,
                                action="skip",
                                current_stage_index=current_stage_index,
                                late_start_grace_seconds=late_start_grace_seconds,
                                probs=probs,
                            )
                            continue
                    if timing_action == "wait":
                        continue
                    if timing_action == "defer":
                        next_stage_index = timing_selection.get("next_stage_index") if isinstance(timing_selection, dict) else None
                        if next_stage_index is None:
                            continue
                        deferred_timing[m.market_id] = DeferredTimingDecision(
                            market_id=m.market_id,
                            market_slug=m.market_slug,
                            event_slug=m.event_slug,
                            series_slug=str(m.series_slug or ""),
                            start_time=m.start_time.isoformat(),
                            end_time=m.end_time.isoformat(),
                            timing_policy_name=model_source,
                            next_stage_index=int(next_stage_index),
                            deferred_at=now.isoformat(),
                        )
                        logger.info(
                            "monitor_timing_policy_defer",
                            extra={
                                "extra": {
                                    "market": m.market_slug,
                                    "market_id": m.market_id,
                                    "timing_policy_name": model_source,
                                    "next_stage_index": int(next_stage_index),
                                    "reason": timing_selection.get("reason") if isinstance(timing_selection, dict) else None,
                                }
                            },
                        )
                        append_timing_policy_event(
                            timing_events_path,
                            mode="monitor-5m",
                            now=now,
                            market=m,
                            seconds_from_start=seconds_from_start,
                            selection=timing_selection,
                            action="defer",
                            current_stage_index=current_stage_index,
                            late_start_grace_seconds=late_start_grace_seconds,
                        )
                        continue
                    if timing_action == "skip":
                        deferred_timing.pop(m.market_id, None)
                        logger.info(
                            "monitor_timing_policy_skip",
                            extra={
                                "extra": {
                                    "market": m.market_slug,
                                    "market_id": m.market_id,
                                    "timing_policy_name": model_source,
                                    "reason": timing_selection.get("reason") if isinstance(timing_selection, dict) else None,
                                }
                            },
                        )
                        append_timing_policy_event(
                            timing_events_path,
                            mode="monitor-5m",
                            now=now,
                            market=m,
                            seconds_from_start=seconds_from_start,
                            selection=timing_selection,
                            action="skip",
                            current_stage_index=current_stage_index,
                            late_start_grace_seconds=late_start_grace_seconds,
                        )
                        continue
                    deferred_timing.pop(m.market_id, None)
                    append_timing_policy_event(
                        timing_events_path,
                        mode="monitor-5m",
                        now=now,
                        market=m,
                        seconds_from_start=seconds_from_start,
                        selection=timing_selection,
                        action="finalize",
                        current_stage_index=current_stage_index,
                        late_start_grace_seconds=late_start_grace_seconds,
                        probs=probs,
                    )
                p_up = float(probs["p_up"])
                p_down = float(probs["p_down"])
                predicted_side = str(probs["predicted_side"])
                confidence = float(probs["confidence"])
                shadow_decision = None
                shadow_error: str | None = None
                try:
                    shadow_decision = decide_trade(
                        market_slug=m.market_slug,
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
                        maker_fill_probability=maker_fill_probability,
                        maker_fill_probability_by_token=maker_fill_probability_by_token(
                            books=books,
                            seconds_to_expiry=sec_to_exp,
                            exec_cfg=exec_cfg,
                        ),
                        maker_ev_advantage_required=maker_ev_advantage_required,
                        allowed_order_types=list(allowed_order_types) if isinstance(allowed_order_types, list) else allowed_order_types,
                        lock_side_to_prediction=bool(exec_cfg.get("lock_side_to_prediction", True)),
                        min_reward_to_risk_ratio=float(exec_cfg.get("min_reward_to_risk_ratio", 0.0)),
                        min_expected_roi_cash=to_optional_float(exec_cfg.get("min_expected_roi_cash")),
                        min_breakeven_margin=to_optional_float(exec_cfg.get("min_breakeven_margin")),
                        sizing_mode=str(exec_cfg.get("sizing_mode", "edge_scaled")),
                        max_entry_price=to_optional_float(exec_cfg.get("max_entry_price")),
                        actionability_calibration=actionability_calibration_cfg(exec_cfg),
                        actionability_context=actionability_context(
                            p_up=float(p_up),
                            proxy_p_up=to_optional_float(probs.get("proxy_p_up")),
                            spot_recent_vol_5m_bps=(
                                float(spot_ctx.spot_recent_vol_5m_bps)
                                if spot_ctx is not None and spot_ctx.spot_recent_vol_5m_bps is not None
                                else None
                            ),
                        ),
                        skip_confidence_band=skip_confidence_band(exec_cfg),
                    )
                except Exception as exc:
                    shadow_error = str(exc)
                confirmation_gate_payload: dict[str, Any] = {}
                if shadow_decision is not None:
                    shadow_decision, confirmation_gate_payload = apply_confirmation_gate(
                        decision=shadow_decision,
                        books=books,
                        candidate_models=candidate_models_payload,
                        gate_cfg=exec_cfg.get("confirmation_gate", {}),
                    )
                if (
                    shadow_decision is not None
                    and shadow_decision.intent is not None
                    and shadow_decision.action == "trade"
                    and str(shadow_decision.intent.order_type).upper() == "MAKER"
                ):
                    selected_shadow_fee_bps = fee_bps_map.get(shadow_decision.intent.token_id)
                    shadow_auditor.place_virtual_maker_order(
                        market_id=m.market_id,
                        token_id=shadow_decision.intent.token_id,
                        prediction_side=predicted_side,
                        order_side=shadow_decision.intent.side,
                        price=shadow_decision.intent.price,
                        size=shadow_decision.intent.size,
                        books=books,
                        taker_fee_bps=selected_shadow_fee_bps,
                        taker_fee_rate=(
                            float(selected_shadow_fee_bps) / 10_000.0
                            if selected_shadow_fee_bps is not None
                            else fee_cfg.taker_fee_rate
                        ),
                        maker_fee_rate=fee_cfg.maker_fee_rate,
                        taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12)),
                        min_fee=float(fee_cfg.min_fee),
                        order_type=shadow_decision.intent.order_type,
                        now=now,
                    )
                adaptive_risk_payload: dict[str, Any] = {}
                shadow_candidate_side = None
                if shadow_decision is not None:
                    shadow_candidate_side = (
                        outcome_side_for_intent(
                            shadow_decision.intent,
                            up_token_id=books.up.token_id,
                            down_token_id=books.down.token_id,
                        )
                        if shadow_decision.intent is not None
                        else None
                    )
                    adaptive_risk_payload = build_monitor_adaptive_risk_shadow_payload(
                        risk_cfg=risk_cfg,
                        paper_cfg=paper_cfg,
                        decision=shadow_decision,
                        books=books,
                        candidate_side=shadow_candidate_side,
                        confidence=confidence,
                        spot_recent_vol_5m_bps=(
                            float(spot_ctx.spot_recent_vol_5m_bps)
                            if spot_ctx is not None and spot_ctx.spot_recent_vol_5m_bps is not None
                            else None
                        ),
                        seconds_to_expiry=sec_to_exp,
                    )

                rec = PendingPrediction(
                    market_id=m.market_id,
                    market_slug=m.market_slug,
                    event_slug=m.event_slug,
                    series_slug=str(m.series_slug or ""),
                    start_time=m.start_time.isoformat(),
                    end_time=m.end_time.isoformat(),
                    predicted_side=predicted_side,
                    p_up=p_up,
                    p_down=p_down,
                    confidence=confidence,
                    created_at=now.isoformat(),
                    seconds_from_window_start=seconds_from_start,
                    seconds_to_expiry=sec_to_exp,
                    schema_version=OUTPUT_SCHEMA_VERSION,
                    question=m.question,
                    resolution_source=m.resolution_source,
                    model_source=str(probs["model_used"]),
                    model_p_up=probs["model_p_up"],
                    market_p_up=float(probs["market_p_up"]),
                    proxy_p_up=float(probs["proxy_p_up"]),
                    blend_model_weight=model_weight if model_source == "adaptive_blend" else None,
                    calibration_shift=calibration_shift,
                    up_best_bid=books.up.best_bid,
                    up_best_ask=books.up.best_ask,
                    up_midpoint=books.up.midpoint,
                    up_spread=books.up.spread,
                    down_best_bid=books.down.best_bid,
                    down_best_ask=books.down.best_ask,
                    down_midpoint=books.down.midpoint,
                    down_spread=books.down.spread,
                    up_topk_imbalance=books.up.topk_imbalance,
                    down_topk_imbalance=books.down.topk_imbalance,
                    up_best_bid_size=books.up.best_bid_size,
                    up_best_ask_size=books.up.best_ask_size,
                    down_best_bid_size=books.down.best_bid_size,
                    down_best_ask_size=books.down.best_ask_size,
                    up_top3_bid_size=books.up.top3_bid_size,
                    up_top3_ask_size=books.up.top3_ask_size,
                    down_top3_bid_size=books.down.top3_bid_size,
                    down_top3_ask_size=books.down.top3_ask_size,
                    **book_depth_payload(books),
                    decision_action=("error" if shadow_decision is None and shadow_error else (shadow_decision.action if shadow_decision is not None else None)),
                    decision_reason=(shadow_error if shadow_error else (shadow_decision.reason if shadow_decision is not None else None)),
                    decision_best_edge=(shadow_decision.best_edge if shadow_decision is not None else None),
                    decision_score_mode=(shadow_decision.score_mode if shadow_decision is not None else None),
                    decision_score_value=(shadow_decision.score_value if shadow_decision is not None else None),
                    decision_expected_roi_cash=(shadow_decision.expected_roi_cash if shadow_decision is not None else None),
                    decision_breakeven_probability=(
                        shadow_decision.breakeven_probability if shadow_decision is not None else None
                    ),
                    decision_breakeven_margin=(shadow_decision.breakeven_margin if shadow_decision is not None else None),
                    decision_fill_probability=(shadow_decision.fill_probability if shadow_decision is not None else None),
                    decision_ev_executable=(shadow_decision.ev_executable if shadow_decision is not None else None),
                    decision_ev_fill=(shadow_decision.ev_fill if shadow_decision is not None else None),
                    decision_cash_required=(shadow_decision.cash_required if shadow_decision is not None else None),
                    decision_calibrated_p_side=(shadow_decision.calibrated_p_side if shadow_decision is not None else None),
                    decision_calibrated_net_edge=(
                        shadow_decision.calibrated_net_edge if shadow_decision is not None else None
                    ),
                    decision_calibrated_expected_roi_cash=(
                        shadow_decision.calibrated_expected_roi_cash if shadow_decision is not None else None
                    ),
                    decision_calibrated_breakeven_margin=(
                        shadow_decision.calibrated_breakeven_margin if shadow_decision is not None else None
                    ),
                    decision_actionability_calibration_applied=(
                        shadow_decision.actionability_calibration_applied if shadow_decision is not None else None
                    ),
                    decision_actionability_calibration_rejected=(
                        shadow_decision.actionability_calibration_rejected if shadow_decision is not None else None
                    ),
                    decision_actionability_calibration_reason=(
                        shadow_decision.actionability_calibration_reason if shadow_decision is not None else None
                    ),
                    decision_actionability_lower_bound_floor=(
                        shadow_decision.actionability_lower_bound_floor if shadow_decision is not None else None
                    ),
                    decision_actionability_lower_bound_sources=(
                        list(shadow_decision.actionability_lower_bound_sources)
                        if shadow_decision is not None and shadow_decision.actionability_lower_bound_sources is not None
                        else None
                    ),
                    decision_order_type=(
                        shadow_decision.intent.order_type if shadow_decision is not None and shadow_decision.intent is not None else None
                    ),
                    decision_expected_edge=(
                        shadow_decision.intent.expected_edge if shadow_decision is not None and shadow_decision.intent is not None else None
                    ),
                    decision_token_id=(
                        shadow_decision.intent.token_id if shadow_decision is not None and shadow_decision.intent is not None else None
                    ),
                    book_quality_ok=quality.ok,
                    book_quality_score=quality.score,
                    book_quality_reason=quality.reason,
                    up_taker_fee_bps=fee_bps_map.get(books.up.token_id),
                    down_taker_fee_bps=fee_bps_map.get(books.down.token_id),
                    oracle_basis_bps=(oracle_basis.basis_bps if oracle_basis is not None else None),
                    oracle_spot_price=(oracle_basis.spot_price if oracle_basis is not None else None),
                    oracle_price=(oracle_basis.oracle_price if oracle_basis is not None else None),
                    oracle_source=oracle_source,
                    oracle_external_feed=oracle_external_feed,
                    spot_price_now=(spot_ctx.spot_price_now if spot_ctx is not None else None),
                    spot_window_open_price=(spot_ctx.spot_window_open_price if spot_ctx is not None else None),
                    spot_return_bps_from_open=(spot_ctx.spot_return_bps_from_open if spot_ctx is not None else None),
                    spot_recent_return_1m_bps=(spot_ctx.spot_recent_return_1m_bps if spot_ctx is not None else None),
                    spot_recent_vol_5m_bps=(spot_ctx.spot_recent_vol_5m_bps if spot_ctx is not None else None),
                    candidate_models=candidate_models_payload,
                    timing_policy_name=(str(probs.get("timing_policy_name")) if probs.get("timing_policy_name") is not None else None),
                    timing_policy_source=(str(probs.get("timing_policy_source")) if probs.get("timing_policy_source") is not None else None),
                    timing_policy_stage_delay=(
                        int(probs.get("timing_policy_stage_delay")) if probs.get("timing_policy_stage_delay") is not None else None
                    ),
                    timing_policy_stage_extra_edge=(
                        float(probs.get("timing_policy_stage_extra_edge")) if probs.get("timing_policy_stage_extra_edge") is not None else None
                    ),
                    timing_policy_reason=(str(probs.get("timing_policy_reason")) if probs.get("timing_policy_reason") is not None else None),
                    timing_policy_entry_mode=(str(probs.get("timing_policy_entry_mode")) if probs.get("timing_policy_entry_mode") is not None else None),
                    adaptive_risk_payload=adaptive_risk_payload or None,
                    confirmation_gate_payload=confirmation_gate_payload or None,
                )
                pending[m.market_id] = rec

                pred_payload = {
                    "ts_utc": now.isoformat(),
                    **asdict(rec),
                    "seconds_from_window_start": seconds_from_start,
                    "seconds_to_expiry": sec_to_exp,
                    "book_quality_flags": quality.flags,
                    "book_quality_metrics": quality.metrics,
                    "p_pre_calibration": probs["p_pre_calibration"],
                    "oracle_spot_price": (oracle_basis.spot_price if oracle_basis is not None else None),
                    "oracle_price": (oracle_basis.oracle_price if oracle_basis is not None else None),
                }
                if adaptive_risk_payload:
                    pred_payload.update(adaptive_risk_payload)
                if confirmation_gate_payload:
                    pred_payload.update(confirmation_gate_payload)
                append_jsonl(predictions_path, pred_payload)
                if adaptive_risk_payload:
                    append_jsonl(
                        adaptive_risk_shadow_path,
                        {
                            "ts_utc": now.isoformat(),
                            "market_id": m.market_id,
                            "market_slug": m.market_slug,
                            "event_slug": m.event_slug,
                            "series_slug": str(m.series_slug or ""),
                            "start_time": m.start_time.isoformat(),
                            "end_time": m.end_time.isoformat(),
                            "predicted_side": predicted_side,
                            "p_up": p_up,
                            "p_down": p_down,
                            "model_source": str(probs["model_used"]),
                            "entry_price": (
                                float(shadow_decision.intent.price)
                                if shadow_decision is not None and shadow_decision.intent is not None
                                else None
                            ),
                            "candidate_side": shadow_candidate_side,
                            **adaptive_risk_payload,
                        },
                    )
                logger.info("monitor_prediction", extra={"extra": pred_payload})

            for market_id, rec in list(pending.items()):
                end_time = parse_iso_utc(rec.end_time)
                if now < end_time + timedelta(seconds=settlement_grace_seconds):
                    continue

                latest_market = gamma.get_market_by_id(market_id)
                if not latest_market:
                    payload = {"market_id": market_id}
                    if gamma.last_error:
                        payload.update(gamma_error_payload(gamma))
                        logger.warning("monitor_missing_market_network_error", extra={"extra": payload})
                    else:
                        logger.info("monitor_missing_market", extra={"extra": payload})
                    continue

                actual_side = extract_resolved_outcome_side(
                    latest_market,
                    min_winner_price=resolution_winner_threshold,
                )
                if actual_side is None:
                    logger.info(
                        "monitor_wait_resolution",
                        extra={
                            "extra": {
                                "market_id": market_id,
                                "market_slug": rec.market_slug,
                                "closed": bool(latest_market.get("closed", False)),
                            }
                        },
                    )
                    continue

                is_correct = actual_side == rec.predicted_side
                resolved_ids.add(market_id)
                shadow_auditor.resolve_markets(market_id, actual_side)
                logger.info("shadow_auditor_summary", extra={"extra": shadow_auditor.get_summary()})
                pending.pop(market_id, None)

                total += 1
                if is_correct:
                    correct += 1
                accuracy = (correct / total) if total > 0 else 0.0

                _rolling_outcomes.append(is_correct)
                _rolling_wr_metrics: dict[str, Any] = {}
                for _lb in _ROLLING_WR_LOOKBACKS:
                    _recent = list(_rolling_outcomes)[-_lb:]
                    if len(_recent) >= _lb:
                        _wr = sum(_recent) / len(_recent)
                        _rolling_wr_metrics[f"rolling_wr_{_lb}"] = round(_wr, 4)
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_52"] = _wr >= 0.52
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_50"] = _wr >= 0.50
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_48"] = _wr >= 0.48
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_45"] = _wr >= 0.45
                    else:
                        _rolling_wr_metrics[f"rolling_wr_{_lb}"] = None
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_52"] = True
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_50"] = True
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_48"] = True
                        _rolling_wr_metrics[f"rolling_wr_{_lb}_would_trade_45"] = True

                result_payload = {
                    "ts_utc": now.isoformat(),
                    **asdict(rec),
                    "question": rec.question or str(latest_market.get("question", "")) or None,
                    "resolution_source": rec.resolution_source or str(latest_market.get("resolutionSource", "")) or None,
                    "resolved_ts_utc": now.isoformat(),
                    "actual_side": actual_side,
                    "is_correct": is_correct,
                    "running_total": total,
                    "running_correct": correct,
                    "running_accuracy": accuracy,
                    **_rolling_wr_metrics,
                }
                append_jsonl(outcomes_path, result_payload)
                logger.info("monitor_resolution", extra={"extra": result_payload})

            summary = build_monitor_summary(
                now=now,
                started_at=started_at,
                pending=pending,
                deferred_timing=deferred_timing,
                resolved_ids=resolved_ids,
                total=total,
                correct=correct,
            )
            write_json(state_path, summary)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("monitor_stopped", extra={"extra": {"reason": "keyboard_interrupt"}})

    end_now = datetime.now(UTC)
    summary = build_monitor_summary(
        now=end_now,
        started_at=started_at,
        pending=pending,
        deferred_timing=deferred_timing,
        resolved_ids=resolved_ids,
        total=total,
        correct=correct,
    )
    write_json(state_path, summary)

    day_path = Path(cfg["paths"]["outputs_dir"]) / f"monitor_5m_summary_{end_now.strftime('%Y%m%d')}.json"
    write_json(day_path, summary)
    return summary


