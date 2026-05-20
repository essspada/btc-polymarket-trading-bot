"""Paper-trading loop: simulates orders against live books with bankroll state.

Mirrors the monitor mode but additionally opens and resolves paper trades
against real fills/midpoints, tracks a bankroll, and persists pending
positions so a restart can resume cleanly. No live orders are ever sent.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.oracles import OracleClient
from src.data.spot_context import SpotContextClient
from src.execution.paper_execution import PaperExecutionEngine, PaperTrade
from src.execution.shadow_auditor import ShadowAuditor
from src.polymarket.book_quality import BookQualityConfig, assess_market_books
from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.clients.gamma_client import GammaClient
from src.polymarket.fees import FeeModelConfig
from src.polymarket.market_discovery import (
    discover_btc_5m_window_markets,
    extract_resolved_outcome_side,
)
from src.polymarket.risk import RiskManager
from src.runtime.persistence import (
    append_jsonl,
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
    build_adaptive_risk_shadow_payload,
)
from src.runtime.sizing import (
    apply_sizing_cap_to_intent,
    apply_stateful_risk_multiplier_to_intent,
    cash_required_for_order,
)
from src.runtime.snapshots import (
    book_depth_payload,
    build_market_books,
)
from src.runtime.state import (
    OUTPUT_SCHEMA_VERSION,
    PendingPaperTrade,
    PendingPrediction,
    count_active_pending_paper_trades,
    load_deferred_timing,
    load_paper_bankroll_state,
    load_pending_paper_trades,
    load_pending_predictions,
    load_running_stats,
)
from src.runtime.timing import (
    apply_runtime_timing_policy,
    prune_deferred_timing,
    risk_window_exposure_multiplier,
)
from src.runtime.utilities import (
    fetch_taker_fee_bps_by_token,
    gamma_error_payload,
    maker_fill_probability_by_token,
    parse_iso_utc,
    resolve_output_path,
    to_optional_float,
)
from src.runtime.x3 import (
    apply_x3_active_to_decision,
    build_x3_shadow_payload,
    outcome_side_for_intent,
    selected_spread_for_side,
)
from src.strategy.calibration import (
    compute_adaptive_model_weight,
    compute_online_bias_shift,
    fit_rolling_platt,
)
from src.strategy.confirmation_gate import apply_confirmation_gate
from src.strategy.model_wrapper import MarkovProxyModel
from src.strategy.signals import decide_trade
from src.strategy.sizing import apply_conservative_size_cap, cap_exposure_by_balance
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
    select_timing_policy_stage,
    timing_policy_entry_mode,
)
from src.strategy.transferred_markov import TransferredMarkovConfig, TransferredMarkovPredictor
from src.utils.logging import build_logger


def run_paper_5m(cfg: dict[str, Any], max_runtime_minutes: float | None = None) -> dict[str, Any]:
    logger = build_logger(f"{cfg['app']['name']}_paper", cfg["paths"]["logs_dir"])
    gamma = GammaClient(cfg["polymarket"]["gamma_base_url"])
    clob = ClobClient(cfg["polymarket"]["clob_base_url"])
    proxy_model = MarkovProxyModel(default_confidence=float(cfg["model"].get("default_confidence", 0.5)))

    model_cfg = cfg.get("model", {})
    model_source = str(model_cfg.get("source", "proxy_orderbook")).strip().lower()
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

    mon_cfg = cfg.get("monitor", {})
    disc_cfg = cfg.get("discovery", {})
    exec_cfg = cfg.get("execution", {})
    risk_cfg = cfg.get("risk", {})
    paper_cfg = cfg.get("paper", {})
    poll_seconds = max(1, int(mon_cfg.get("poll_seconds", 10)))
    timing_runtime_cfg = model_cfg.get("timing_policy_runtime", {}) if isinstance(model_cfg, dict) else {}
    allow_late_fresh_start = bool(timing_runtime_cfg.get("allow_late_fresh_start", False))
    late_start_grace_seconds = max(0.0, float(timing_runtime_cfg.get("late_start_grace_seconds", poll_seconds)))
    settlement_grace_seconds = max(0, int(mon_cfg.get("settlement_grace_seconds", 45)))
    lookahead_windows = max(1, int(mon_cfg.get("lookahead_windows", 12)))
    lookback_windows = max(0, int(mon_cfg.get("lookback_windows", 1)))
    min_start_delay_seconds = max(
        0.0,
        float(paper_cfg.get("min_start_delay_seconds", mon_cfg.get("min_start_delay_seconds", 0.0))),
    )
    max_start_delay_seconds = max(
        min_start_delay_seconds,
        float(paper_cfg.get("max_start_delay_seconds", mon_cfg.get("max_start_delay_seconds", 75))),
    )
    settle_only = bool(paper_cfg.get("settle_only", False))
    resolution_winner_threshold = float(mon_cfg.get("resolution_winner_threshold", 0.99))
    maker_fill_probability = float(exec_cfg.get("maker_fill_probability", 0.65))
    maker_ev_advantage_required = float(exec_cfg.get("maker_ev_advantage_required", 0.0005))
    allowed_order_types = exec_cfg.get("allowed_order_types", ["maker", "taker"])

    runtime_limit = max_runtime_minutes
    if runtime_limit is None:
        runtime_limit = float(paper_cfg.get("max_runtime_minutes", mon_cfg.get("max_runtime_minutes", 0)))
    runtime_limit = max(0.0, float(runtime_limit))

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    predictions_path = resolve_output_path(
        outputs_dir, str(paper_cfg.get("predictions_file", "paper_predictions_5m.jsonl"))
    )
    outcomes_path = resolve_output_path(outputs_dir, str(paper_cfg.get("outcomes_file", "paper_outcomes_5m.jsonl")))
    trades_path = resolve_output_path(outputs_dir, str(paper_cfg.get("trades_file", "paper_trades_5m.jsonl")))
    x3_shadow_path = resolve_output_path(
        outputs_dir,
        str(paper_cfg.get("x3_shadow_file", "paper_x3_shadow_decisions_5m.jsonl")),
    )
    adaptive_risk_shadow_path = resolve_output_path(
        outputs_dir,
        str(paper_cfg.get("adaptive_risk_shadow_file", "paper_adaptive_risk_shadow_decisions_5m.jsonl")),
    )
    state_path = resolve_output_path(outputs_dir, str(paper_cfg.get("state_file", "paper_5m_state.json")))
    timing_events_path = resolve_output_path(
        outputs_dir,
        str(paper_cfg.get("timing_events_file", "paper_timing_events_5m.jsonl")),
    )
    paper_initial_balance = float(paper_cfg.get("initial_balance_usd", 100.0))

    fee_cfg = FeeModelConfig(**cfg["fees"])
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
    risk = RiskManager(
        max_exposure_per_window_usd=float(risk_cfg["max_exposure_per_window_usd"]),
        max_daily_loss_usd=float(risk_cfg["max_daily_loss_usd"]),
        cooldown_after_loss_streak=int(risk_cfg["cooldown_after_loss_streak"]),
        cooldown_windows=int(risk_cfg["cooldown_windows"]),
        max_open_positions=int(risk_cfg["max_open_positions"]),
        drift_enabled=bool(risk_cfg.get("drift_enabled", False)),
        drift_lookback=int(risk_cfg.get("drift_lookback", 50)),
        drift_min_samples=int(risk_cfg.get("drift_min_samples", 30)),
        drift_min_accuracy=float(risk_cfg.get("drift_min_accuracy", 0.46)),
        drift_cooldown_windows=int(risk_cfg.get("drift_cooldown_windows", 6)),
        stateful_regime=risk_cfg.get("stateful_regime", {}),
    )
    paper_engine = PaperExecutionEngine(
        maker_fill_floor=float(exec_cfg.get("maker_fill_floor", 0.05)),
        maker_fill_cap=float(exec_cfg.get("maker_fill_cap", 0.9)),
        maker_fill_base=float(exec_cfg.get("maker_fill_base", 0.78)),
        maker_fill_spread_penalty=float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        maker_fill_late_penalty_90=float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        maker_fill_late_penalty_45=float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    )

    pending = load_pending_predictions(state_path)
    shadow_auditor = ShadowAuditor()
    deferred_timing = load_deferred_timing(state_path)
    pending_trades = load_pending_paper_trades(state_path)
    resolved_ids, total, correct = load_running_stats(outcomes_path)
    for market_id in list(pending.keys()):
        if market_id in resolved_ids:
            pending.pop(market_id, None)
    for market_id in list(pending_trades.keys()):
        if market_id in resolved_ids:
            pending_trades.pop(market_id, None)
    prune_deferred_timing(
        deferred=deferred_timing,
        resolved_ids=resolved_ids,
        now=datetime.now(UTC),
        settlement_grace_seconds=settlement_grace_seconds,
    )
    paper_bankroll = load_paper_bankroll_state(
        state_path=state_path,
        outcomes_path=outcomes_path,
        pending_trades=pending_trades,
        initial_balance=paper_initial_balance,
        fee_cfg=fee_cfg,
        exec_cfg=exec_cfg,
    )
    paper_peak_cash = max(float(paper_bankroll.initial_balance), float(paper_bankroll.available_cash))
    risk.state.open_positions = count_active_pending_paper_trades(pending_trades)

    started_at = datetime.now(UTC)
    summary: dict[str, Any] = {}

    logger.info(
        "paper_started",
        extra={
            "extra": {
                "poll_seconds": poll_seconds,
                "lookahead_windows": lookahead_windows,
                "lookback_windows": lookback_windows,
                "min_start_delay_seconds": min_start_delay_seconds,
                "max_start_delay_seconds": max_start_delay_seconds,
                "settle_only": settle_only,
                "runtime_limit_minutes": runtime_limit,
                "model_source": model_source,
                "timing_policy_active": bool(timing_policy_stages),
                "allow_late_fresh_start": allow_late_fresh_start,
                "late_start_grace_seconds": late_start_grace_seconds,
                "existing_resolved": len(resolved_ids),
                "existing_pending_predictions": len(pending),
                "existing_deferred_timing": len(deferred_timing),
                "existing_pending_trades": len(pending_trades),
                "paper_initial_balance_usd": paper_bankroll.initial_balance,
                "paper_available_cash_usd": paper_bankroll.available_cash,
                "paper_reserved_cash_usd": paper_bankroll.reserved_cash,
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

            if not settle_only:
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
                if not discovered and gamma.last_error:
                    logger.warning(
                        "paper_discovery_network_error",
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
                        if m.market_id in pending or m.market_id in pending_trades:
                            if now < m.end_time:
                                books, _book_error = build_market_books(clob, m)
                                if books is not None:
                                    shadow_auditor.on_orderbook_update(m.market_id, books, now=now)
                            continue
                        if now < m.start_time or now >= m.end_time:
                            continue

                    seconds_from_start = (now - m.start_time).total_seconds()
                    deferred_item = deferred_timing.get(m.market_id)
                    current_stage_index = deferred_item.next_stage_index if deferred_item is not None else None
                    if timing_policy_stages:
                        selection_probe = select_timing_policy_stage(
                            stages=timing_policy_stages,
                            seconds_from_start=seconds_from_start,
                            candidate_models_payload=None,
                            min_edge_to_trade=float(exec_cfg.get("min_edge_to_trade", 0.004)),
                            current_stage_index=current_stage_index,
                        )
                        if str(selection_probe.get("action") or "") == "wait":
                            continue
                    else:
                        if seconds_from_start < min_start_delay_seconds:
                            continue
                        if seconds_from_start > max_start_delay_seconds:
                            continue

                    sec_to_exp = max(0.0, (m.end_time - now).total_seconds())
                    spot_ctx = spot_client.fetch_window_context(now=now, window_start=m.start_time) if spot_client is not None else None
                    books, book_error = build_market_books(clob, m)
                    if books is None:
                        logger.info(
                            "paper_skip_no_book",
                            extra={"extra": {"market": m.market_slug, "market_id": m.market_id, "error": book_error}},
                        )
                        continue
                    shadow_auditor.on_orderbook_update(m.market_id, books, now=now)

                    quality = assess_market_books(books=books, cfg=book_quality_cfg)
                    if not quality.ok:
                        logger.info(
                            "paper_skip_book_quality",
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
                                        "paper_timing_policy_defer",
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
                                        mode="paper-5m",
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
                                    "paper_timing_policy_skip",
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
                                    mode="paper-5m",
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
                                "paper_timing_policy_defer",
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
                                mode="paper-5m",
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
                                "paper_timing_policy_skip",
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
                                mode="paper-5m",
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
                            mode="paper-5m",
                            now=now,
                            market=m,
                            seconds_from_start=seconds_from_start,
                            selection=timing_selection,
                            action="finalize",
                            current_stage_index=current_stage_index,
                            late_start_grace_seconds=late_start_grace_seconds,
                            probs=probs,
                        )
                    capped_exposure_usd = cap_exposure_by_balance(
                        max_exposure_usd=float(risk_cfg["max_exposure_per_window_usd"]),
                        balance_usd=float(paper_bankroll.available_cash),
                        max_balance_fraction_per_trade=float(risk_cfg.get("max_balance_fraction_per_trade", 1.0)),
                    )
                    capped_exposure_usd *= risk_window_exposure_multiplier(now=now, risk_cfg=risk_cfg)
                    decision = decide_trade(
                        market_slug=m.market_slug,
                        books=books,
                        p_up=float(probs["p_up"]),
                        min_edge_to_trade=float(exec_cfg["min_edge_to_trade"]),
                        min_edge_for_taker=float(exec_cfg["min_edge_for_taker"]),
                        max_exposure_usd=float(capped_exposure_usd),
                        maker_preference=bool(exec_cfg.get("maker_preference", True)),
                        fee_cfg=fee_cfg,
                        taker_slippage_bps=float(exec_cfg["slippage_bps_taker"]),
                        maker_slippage_bps=float(exec_cfg["slippage_bps_maker"]),
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
                            p_up=float(probs["p_up"]),
                            proxy_p_up=to_optional_float(probs.get("proxy_p_up")),
                            spot_recent_vol_5m_bps=(
                                float(spot_ctx.spot_recent_vol_5m_bps)
                                if spot_ctx is not None and spot_ctx.spot_recent_vol_5m_bps is not None
                                else None
                            ),
                        ),
                        skip_confidence_band=skip_confidence_band(exec_cfg),
                    )
                    decision, confirmation_gate_payload = apply_confirmation_gate(
                        decision=decision,
                        books=books,
                        candidate_models=candidate_models_payload,
                        gate_cfg=exec_cfg.get("confirmation_gate", {}),
                    )
                    raw_cash_required = 0.0
                    cash_required = 0.0
                    stateful_payload: dict[str, Any] = {
                        "stateful_regime_enabled": bool(risk_cfg.get("stateful_regime", {}).get("enabled", False)),
                        "stateful_multiplier_applied": False,
                        "stateful_multiplier": 1.0,
                        "stateful_reason": "stateful_disabled",
                    }
                    _, _, sizing_cap_payload = apply_conservative_size_cap(
                        raw_size=0.0,
                        raw_cash_required=0.0,
                        base_exposure_usd=float(capped_exposure_usd),
                        sizing_cap_cfg=risk_cfg.get("sizing_cap", {}),
                    )
                    if decision.intent is not None:
                        selected_taker_fee_bps = fee_bps_map.get(decision.intent.token_id)
                        raw_cash_required = cash_required_for_order(
                            price=float(decision.intent.price),
                            size=float(decision.intent.size),
                            order_type=str(decision.intent.order_type),
                            fee_cfg=fee_cfg,
                            taker_slippage_bps=float(exec_cfg["slippage_bps_taker"]),
                            maker_slippage_bps=float(exec_cfg["slippage_bps_maker"]),
                            taker_fee_bps=selected_taker_fee_bps,
                        )
                        stateful_multiplier, stateful_reason = risk.stateful_trade_multiplier(
                            available_cash=float(paper_bankroll.available_cash),
                            peak_cash=float(paper_peak_cash),
                            ts=now,
                        )
                        decision, stateful_cash_required, stateful_payload = apply_stateful_risk_multiplier_to_intent(
                            decision=decision,
                            raw_cash_required=float(raw_cash_required),
                            multiplier=float(stateful_multiplier),
                        )
                        stateful_payload["stateful_reason"] = str(stateful_reason)
                        decision.intent, cash_required, sizing_cap_payload = apply_sizing_cap_to_intent(
                            intent=decision.intent,
                            raw_cash_required=stateful_cash_required,
                            base_exposure_usd=float(capped_exposure_usd),
                            risk_cfg=risk_cfg,
                        )

                    x3_candidate_side = (
                        outcome_side_for_intent(
                            decision.intent,
                            up_token_id=books.up.token_id,
                            down_token_id=books.down.token_id,
                        )
                        if decision.intent is not None
                        else None
                    )
                    x3_active_payload: dict[str, Any] = {}
                    if decision.intent is not None:
                        decision, cash_required, x3_active_payload = apply_x3_active_to_decision(
                            risk_cfg=risk_cfg,
                            decision=decision,
                            candidate_cash_required=float(cash_required),
                            raw_cash_required=float(raw_cash_required),
                            sizing_cap_payload=sizing_cap_payload,
                            candidate_side=x3_candidate_side,
                        )

                    x3_shadow_payload = build_x3_shadow_payload(
                        risk_cfg=risk_cfg,
                        decision=decision,
                        candidate_cash_required=float(cash_required),
                        raw_cash_required=float(raw_cash_required),
                        sizing_cap_payload=sizing_cap_payload,
                        candidate_side=x3_candidate_side,
                    )
                    x3_payload = x3_active_payload or x3_shadow_payload
                    current_equity = float(paper_bankroll.available_cash + paper_bankroll.reserved_cash)
                    peak_equity = max(float(paper_peak_cash), current_equity, float(paper_bankroll.initial_balance))
                    adaptive_risk_payload = build_adaptive_risk_shadow_payload(
                        risk_cfg=risk_cfg,
                        decision=decision,
                        candidate_cash_required=float(cash_required if decision.intent is not None else 0.0),
                        current_equity_usd=float(current_equity),
                        available_cash_usd=float(paper_bankroll.available_cash),
                        reserved_cash_usd=float(paper_bankroll.reserved_cash),
                        peak_equity_usd=float(peak_equity),
                        daily_pnl=float(risk.state.daily_pnl),
                        loss_streak=int(risk.state.loss_streak),
                        consecutive_wins=int(risk.state.consecutive_wins),
                        cooldown_left=int(risk.state.cooldown_left),
                        recent_accuracy=float(risk.state.recent_accuracy),
                        open_positions=int(risk.state.open_positions),
                        max_open_positions=int(risk.max_open_positions),
                        risk_budget_cash_usd=float(capped_exposure_usd),
                        confidence=float(probs["confidence"]),
                        spread_norm=selected_spread_for_side(books, x3_candidate_side),
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
                        predicted_side=str(probs["predicted_side"]),
                        p_up=float(probs["p_up"]),
                        p_down=float(probs["p_down"]),
                        confidence=float(probs["confidence"]),
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
                        decision_action=decision.action,
                        decision_reason=decision.reason,
                        decision_best_edge=decision.best_edge,
                        decision_score_mode=decision.score_mode,
                        decision_score_value=decision.score_value,
                        decision_expected_roi_cash=decision.expected_roi_cash,
                        decision_breakeven_probability=decision.breakeven_probability,
                        decision_breakeven_margin=decision.breakeven_margin,
                        decision_fill_probability=decision.fill_probability,
                        decision_ev_executable=decision.ev_executable,
                        decision_ev_fill=decision.ev_fill,
                        decision_cash_required=decision.cash_required,
                        decision_calibrated_p_side=decision.calibrated_p_side,
                        decision_calibrated_net_edge=decision.calibrated_net_edge,
                        decision_calibrated_expected_roi_cash=decision.calibrated_expected_roi_cash,
                        decision_calibrated_breakeven_margin=decision.calibrated_breakeven_margin,
                        decision_actionability_calibration_applied=decision.actionability_calibration_applied,
                        decision_actionability_calibration_rejected=decision.actionability_calibration_rejected,
                        decision_actionability_calibration_reason=decision.actionability_calibration_reason,
                        decision_actionability_lower_bound_floor=decision.actionability_lower_bound_floor,
                        decision_actionability_lower_bound_sources=(
                            list(decision.actionability_lower_bound_sources)
                            if decision.actionability_lower_bound_sources is not None
                            else None
                        ),
                        decision_order_type=decision.intent.order_type if decision.intent is not None else None,
                        decision_expected_edge=decision.intent.expected_edge if decision.intent is not None else None,
                        decision_token_id=decision.intent.token_id if decision.intent is not None else None,
                        book_quality_ok=quality.ok,
                        book_quality_score=quality.score,
                        book_quality_reason=quality.reason,
                        up_taker_fee_bps=fee_bps_map.get(books.up.token_id),
                        down_taker_fee_bps=fee_bps_map.get(books.down.token_id),
                        oracle_basis_bps=(oracle_basis.basis_bps if oracle_basis is not None else None),
                        oracle_spot_price=(oracle_basis.spot_price if oracle_basis is not None else None),
                        oracle_price=(oracle_basis.oracle_price if oracle_basis is not None else None),
                        oracle_source=(
                            "external"
                            if bool(oracle_cfg.get("oracle_price_url"))
                            else ("spot_fallback" if oracle_client is not None else "disabled")
                        ),
                        oracle_external_feed=bool(oracle_cfg.get("oracle_price_url")),
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
                        sizing_cap_enabled=bool(sizing_cap_payload.get("sizing_cap_enabled", False)),
                        sizing_cap_applied=bool(sizing_cap_payload.get("sizing_cap_applied", False)),
                        sizing_cap_reason=(
                            str(sizing_cap_payload.get("sizing_cap_reason"))
                            if sizing_cap_payload.get("sizing_cap_reason") is not None
                            else None
                        ),
                        sizing_cap_ratio=(
                            float(sizing_cap_payload.get("sizing_cap_ratio"))
                            if sizing_cap_payload.get("sizing_cap_ratio") is not None
                            else None
                        ),
                        sizing_cap_below_min_trade=bool(sizing_cap_payload.get("sizing_cap_below_min_trade", False)),
                        sizing_cap_raw_size=(
                            float(sizing_cap_payload.get("sizing_cap_raw_size"))
                            if sizing_cap_payload.get("sizing_cap_raw_size") is not None
                            else None
                        ),
                        sizing_cap_capped_size=(
                            float(sizing_cap_payload.get("sizing_cap_capped_size"))
                            if sizing_cap_payload.get("sizing_cap_capped_size") is not None
                            else None
                        ),
                        sizing_cap_raw_cash_required=(
                            float(sizing_cap_payload.get("sizing_cap_raw_cash_required"))
                            if sizing_cap_payload.get("sizing_cap_raw_cash_required") is not None
                            else None
                        ),
                        sizing_cap_capped_cash_required=(
                            float(sizing_cap_payload.get("sizing_cap_capped_cash_required"))
                            if sizing_cap_payload.get("sizing_cap_capped_cash_required") is not None
                            else None
                        ),
                        sizing_cap_min_trade_usd_after_cap=(
                            float(sizing_cap_payload.get("sizing_cap_min_trade_usd_after_cap"))
                            if sizing_cap_payload.get("sizing_cap_min_trade_usd_after_cap") is not None
                            else None
                        ),
                        sizing_cap_max_trade_usd=(
                            float(sizing_cap_payload.get("sizing_cap_max_trade_usd"))
                            if sizing_cap_payload.get("sizing_cap_max_trade_usd") is not None
                            else None
                        ),
                        sizing_cap_max_fraction_of_base_exposure=(
                            float(sizing_cap_payload.get("sizing_cap_max_fraction_of_base_exposure"))
                            if sizing_cap_payload.get("sizing_cap_max_fraction_of_base_exposure") is not None
                            else None
                        ),
                        sizing_cap_base_exposure_usd=(
                            float(sizing_cap_payload.get("sizing_cap_base_exposure_usd"))
                            if sizing_cap_payload.get("sizing_cap_base_exposure_usd") is not None
                            else None
                        ),
                        sizing_cap_effective_cap_usd=(
                            float(sizing_cap_payload.get("sizing_cap_effective_cap_usd"))
                            if sizing_cap_payload.get("sizing_cap_effective_cap_usd") is not None
                            else None
                        ),
                        x3_payload=x3_payload or None,
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
                    if x3_payload:
                        pred_payload.update(x3_payload)
                    if adaptive_risk_payload:
                        pred_payload.update(adaptive_risk_payload)
                    if confirmation_gate_payload:
                        pred_payload.update(confirmation_gate_payload)
                    pred_payload.update(stateful_payload)
                    append_jsonl(predictions_path, pred_payload)
                    if x3_shadow_payload:
                        append_jsonl(
                            x3_shadow_path,
                            {
                                "ts_utc": now.isoformat(),
                                "market_id": m.market_id,
                                "market_slug": m.market_slug,
                                "event_slug": m.event_slug,
                                "series_slug": str(m.series_slug or ""),
                                "start_time": m.start_time.isoformat(),
                                "end_time": m.end_time.isoformat(),
                                "predicted_side": str(probs["predicted_side"]),
                                "p_up": float(probs["p_up"]),
                                "p_down": float(probs["p_down"]),
                                "model_source": str(probs["model_used"]),
                                "entry_price": (
                                    float(decision.intent.price)
                                    if decision.intent is not None
                                    else None
                                ),
                                "candidate_side": (
                                    str(decision.intent.side)
                                    if decision.intent is not None
                                    else None
                                ),
                                **x3_shadow_payload,
                            },
                        )
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
                                "predicted_side": str(probs["predicted_side"]),
                                "p_up": float(probs["p_up"]),
                                "p_down": float(probs["p_down"]),
                                "model_source": str(probs["model_used"]),
                                "entry_price": (
                                    float(decision.intent.price)
                                    if decision.intent is not None
                                    else None
                                ),
                                "candidate_side": x3_candidate_side,
                                **adaptive_risk_payload,
                            },
                        )

                    if decision.intent is None:
                        logger.info(
                            "paper_no_trade",
                            extra={
                                "extra": {
                                    "market": m.market_slug,
                                    "market_id": m.market_id,
                                    "reason": decision.reason,
                                    "p_up": decision.p_up,
                                    **confirmation_gate_payload,
                                    **x3_payload,
                                    **adaptive_risk_payload,
                                }
                            },
                        )
                        continue

                    if str(decision.intent.order_type).upper() == "MAKER":
                        selected_shadow_fee_bps = fee_bps_map.get(decision.intent.token_id)
                        shadow_auditor.place_virtual_maker_order(
                            market_id=m.market_id,
                            token_id=decision.intent.token_id,
                            prediction_side=str(probs["predicted_side"]),
                            order_side=decision.intent.side,
                            price=decision.intent.price,
                            size=decision.intent.size,
                            books=books,
                            taker_fee_bps=selected_shadow_fee_bps,
                            taker_fee_rate=(
                                float(selected_shadow_fee_bps) / 10_000.0
                                if selected_shadow_fee_bps is not None
                                else fee_cfg.taker_fee_rate
                            ),
                            maker_fee_rate=fee_cfg.maker_fee_rate,
                            taker_slippage_bps=float(exec_cfg["slippage_bps_taker"]),
                            min_fee=float(fee_cfg.min_fee),
                            order_type=decision.intent.order_type,
                            now=now,
                        )

                    spread_ref = books.up.spread if decision.intent.token_id == books.up.token_id else books.down.spread
                    paper_trade: PaperTrade = paper_engine.open_trade(
                        intent=decision.intent,
                        market_id=m.market_id,
                        created_at=now,
                        up_token_id=books.up.token_id,
                        down_token_id=books.down.token_id,
                        spread=spread_ref,
                        seconds_to_expiry=sec_to_exp,
                        taker_fee_bps=fee_bps_map.get(decision.intent.token_id),
                    )
                    if paper_trade.filled:
                        if cash_required > float(paper_bankroll.available_cash) + 1e-9:
                            logger.info(
                                "paper_cash_block",
                                extra={
                                    "extra": {
                                        "market": m.market_slug,
                                        "market_id": m.market_id,
                                        "raw_cash_required": float(raw_cash_required),
                                        "cash_required": cash_required,
                                        "available_cash": paper_bankroll.available_cash,
                                        **confirmation_gate_payload,
                                        **stateful_payload,
                                        **sizing_cap_payload,
                                        **adaptive_risk_payload,
                                    }
                                },
                            )
                            continue

                        ok, why = risk.can_trade(now, exposure_usd=cash_required)
                        if not ok:
                            logger.info(
                                "paper_risk_block",
                                extra={
                                    "extra": {
                                        "market": m.market_slug,
                                        "market_id": m.market_id,
                                        "reason": why,
                                        "raw_cash_required": float(raw_cash_required),
                                        "cash_required": float(cash_required),
                                        **confirmation_gate_payload,
                                        **stateful_payload,
                                        **sizing_cap_payload,
                                        **adaptive_risk_payload,
                                    }
                                },
                            )
                            continue

                        risk.on_trade_open()
                        paper_bankroll.available_cash = max(0.0, float(paper_bankroll.available_cash) - float(cash_required))
                        paper_bankroll.reserved_cash = max(0.0, float(paper_bankroll.reserved_cash) + float(cash_required))
                        paper_peak_cash = max(float(paper_peak_cash), float(paper_bankroll.available_cash))
                    paper_bankroll.opened_trades += 1
                    pending_trades[m.market_id] = PendingPaperTrade(
                        trade_id=paper_trade.trade_id,
                        market_id=m.market_id,
                        market_slug=m.market_slug,
                        event_slug=m.event_slug,
                        series_slug=str(m.series_slug or ""),
                        start_time=m.start_time.isoformat(),
                        end_time=m.end_time.isoformat(),
                        created_at=now.isoformat(),
                        schema_version=OUTPUT_SCHEMA_VERSION,
                        question=m.question,
                        resolution_source=m.resolution_source,
                        predicted_side=str(probs["predicted_side"]),
                        p_up=float(probs["p_up"]),
                        p_down=float(probs["p_down"]),
                        model_source=str(probs["model_used"]),
                        decision_reason=decision.reason,
                        token_id=paper_trade.token_id,
                        order_type=paper_trade.order_type,
                        fill_price=paper_trade.fill_price,
                        fill_size=paper_trade.fill_size,
                        filled=paper_trade.filled,
                        fill_probability=paper_trade.fill_probability,
                        expected_edge=paper_trade.expected_edge,
                        expected_roi_cash=decision.expected_roi_cash,
                        breakeven_probability=decision.breakeven_probability,
                        breakeven_margin=decision.breakeven_margin,
                        decision_score_mode=decision.score_mode,
                        decision_score_value=decision.score_value,
                        decision_ev_executable=decision.ev_executable,
                        decision_ev_fill=decision.ev_fill,
                        decision_cash_required=decision.cash_required,
                        decision_calibrated_p_side=decision.calibrated_p_side,
                        decision_calibrated_net_edge=decision.calibrated_net_edge,
                        decision_calibrated_expected_roi_cash=decision.calibrated_expected_roi_cash,
                        decision_calibrated_breakeven_margin=decision.calibrated_breakeven_margin,
                        decision_actionability_calibration_applied=decision.actionability_calibration_applied,
                        decision_actionability_calibration_rejected=decision.actionability_calibration_rejected,
                        decision_actionability_calibration_reason=decision.actionability_calibration_reason,
                        decision_actionability_lower_bound_floor=decision.actionability_lower_bound_floor,
                        decision_actionability_lower_bound_sources=(
                            list(decision.actionability_lower_bound_sources)
                            if decision.actionability_lower_bound_sources is not None
                            else None
                        ),
                        up_token_id=books.up.token_id,
                        down_token_id=books.down.token_id,
                        up_best_ask=books.up.best_ask,
                        down_best_ask=books.down.best_ask,
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
                        up_taker_fee_bps=fee_bps_map.get(books.up.token_id),
                        down_taker_fee_bps=fee_bps_map.get(books.down.token_id),
                        selected_taker_fee_bps=fee_bps_map.get(decision.intent.token_id),
                        oracle_basis_bps=(oracle_basis.basis_bps if oracle_basis is not None else None),
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
                        sizing_cap_enabled=bool(sizing_cap_payload.get("sizing_cap_enabled", False)),
                        sizing_cap_applied=bool(sizing_cap_payload.get("sizing_cap_applied", False)),
                        sizing_cap_reason=(
                            str(sizing_cap_payload.get("sizing_cap_reason"))
                            if sizing_cap_payload.get("sizing_cap_reason") is not None
                            else None
                        ),
                        sizing_cap_ratio=(
                            float(sizing_cap_payload.get("sizing_cap_ratio"))
                            if sizing_cap_payload.get("sizing_cap_ratio") is not None
                            else None
                        ),
                        sizing_cap_below_min_trade=bool(sizing_cap_payload.get("sizing_cap_below_min_trade", False)),
                        sizing_cap_raw_size=(
                            float(sizing_cap_payload.get("sizing_cap_raw_size"))
                            if sizing_cap_payload.get("sizing_cap_raw_size") is not None
                            else None
                        ),
                        sizing_cap_capped_size=(
                            float(sizing_cap_payload.get("sizing_cap_capped_size"))
                            if sizing_cap_payload.get("sizing_cap_capped_size") is not None
                            else None
                        ),
                        sizing_cap_raw_cash_required=(
                            float(sizing_cap_payload.get("sizing_cap_raw_cash_required"))
                            if sizing_cap_payload.get("sizing_cap_raw_cash_required") is not None
                            else None
                        ),
                        sizing_cap_capped_cash_required=(
                            float(sizing_cap_payload.get("sizing_cap_capped_cash_required"))
                            if sizing_cap_payload.get("sizing_cap_capped_cash_required") is not None
                            else None
                        ),
                        sizing_cap_min_trade_usd_after_cap=(
                            float(sizing_cap_payload.get("sizing_cap_min_trade_usd_after_cap"))
                            if sizing_cap_payload.get("sizing_cap_min_trade_usd_after_cap") is not None
                            else None
                        ),
                        sizing_cap_max_trade_usd=(
                            float(sizing_cap_payload.get("sizing_cap_max_trade_usd"))
                            if sizing_cap_payload.get("sizing_cap_max_trade_usd") is not None
                            else None
                        ),
                        sizing_cap_max_fraction_of_base_exposure=(
                            float(sizing_cap_payload.get("sizing_cap_max_fraction_of_base_exposure"))
                            if sizing_cap_payload.get("sizing_cap_max_fraction_of_base_exposure") is not None
                            else None
                        ),
                        sizing_cap_base_exposure_usd=(
                            float(sizing_cap_payload.get("sizing_cap_base_exposure_usd"))
                            if sizing_cap_payload.get("sizing_cap_base_exposure_usd") is not None
                            else None
                        ),
                        sizing_cap_effective_cap_usd=(
                            float(sizing_cap_payload.get("sizing_cap_effective_cap_usd"))
                            if sizing_cap_payload.get("sizing_cap_effective_cap_usd") is not None
                            else None
                        ),
                        cash_required=float(cash_required),
                        x3_payload=x3_payload or None,
                        adaptive_risk_payload=adaptive_risk_payload or None,
                        confirmation_gate_payload=confirmation_gate_payload or None,
                    )
                    append_jsonl(
                        trades_path,
                        {
                            "ts_utc": now.isoformat(),
                            "event": "open",
                            **asdict(pending_trades[m.market_id]),
                        },
                    )
                    logger.info(
                        "paper_trade_open",
                        extra={"extra": {"market": m.market_slug, "market_id": m.market_id, **asdict(pending_trades[m.market_id])}},
                    )

            for market_id, rec in list(pending.items()):
                end_time = parse_iso_utc(rec.end_time)
                if now < end_time + timedelta(seconds=settlement_grace_seconds):
                    continue

                latest_market = gamma.get_market_by_id(market_id)
                if not latest_market:
                    payload = {"market_id": market_id}
                    if gamma.last_error:
                        payload.update(gamma_error_payload(gamma))
                        logger.warning("paper_missing_market_network_error", extra={"extra": payload})
                    else:
                        logger.info("paper_missing_market", extra={"extra": payload})
                    continue

                actual_side = extract_resolved_outcome_side(
                    latest_market,
                    min_winner_price=resolution_winner_threshold,
                )
                if actual_side is None:
                    logger.info(
                        "paper_wait_resolution",
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
                risk.on_outcome(is_correct)
                resolved_ids.add(market_id)
                shadow_auditor.resolve_markets(market_id, actual_side)
                logger.info("shadow_auditor_summary", extra={"extra": shadow_auditor.get_summary()})
                pending.pop(market_id, None)

                total += 1
                if is_correct:
                    correct += 1
                accuracy = (correct / total) if total > 0 else 0.0

                trade = pending_trades.pop(market_id, None)
                settlement_payload: dict[str, Any] = {
                    "trade_id": None,
                    "trade_filled": False,
                    "trade_order_type": None,
                    "trade_fill_size": 0.0,
                    "trade_net_pnl": 0.0,
                    "trade_fee": 0.0,
                    "trade_slippage": 0.0,
                }
                if trade is not None:
                    settlement = paper_engine.settle_trade(
                        trade=PaperTrade(
                            trade_id=trade.trade_id,
                            market_id=trade.market_id,
                            market_slug=trade.market_slug,
                            token_id=trade.token_id,
                            order_type=trade.order_type,
                            side="BUY",
                            fill_price=trade.fill_price,
                            fill_size=trade.fill_size,
                            expected_edge=trade.expected_edge,
                            filled=trade.filled,
                            fill_probability=trade.fill_probability,
                            created_at=trade.created_at,
                            up_token_id=trade.up_token_id,
                            down_token_id=trade.down_token_id,
                            taker_fee_bps=trade.up_taker_fee_bps if trade.token_id == trade.up_token_id else trade.down_taker_fee_bps,
                        ),
                        settled_at=now,
                        actual_side=actual_side,
                        fee_cfg=fee_cfg,
                        taker_slippage_bps=float(exec_cfg["slippage_bps_taker"]),
                        maker_slippage_bps=float(exec_cfg["slippage_bps_maker"]),
                    )
                    paper_bankroll.settled_trades += 1
                    if settlement.filled:
                        risk.on_trade_close(settlement.net_pnl)
                        paper_bankroll.reserved_cash = max(
                            0.0,
                            float(paper_bankroll.reserved_cash) - float(getattr(trade, "cash_required", 0.0) or 0.0),
                        )
                        paper_bankroll.available_cash = max(
                            0.0,
                            float(paper_bankroll.available_cash)
                            + float(getattr(trade, "cash_required", 0.0) or 0.0)
                            + float(settlement.net_pnl),
                        )
                        paper_peak_cash = max(float(paper_peak_cash), float(paper_bankroll.available_cash))
                        paper_bankroll.realized_pnl += float(settlement.net_pnl)
                        paper_bankroll.realized_fees += float(settlement.fee)
                        paper_bankroll.realized_slippage += float(settlement.slippage)
                    settlement_payload = {
                        "trade_id": settlement.trade_id,
                        "trade_filled": settlement.filled,
                        "trade_order_type": trade.order_type,
                        "trade_fill_size": trade.fill_size,
                        "trade_net_pnl": settlement.net_pnl,
                        "trade_fee": settlement.fee,
                        "trade_slippage": settlement.slippage,
                    }
                    append_jsonl(
                        trades_path,
                        {
                            "ts_utc": now.isoformat(),
                            "schema_version": OUTPUT_SCHEMA_VERSION,
                            "resolved_ts_utc": now.isoformat(),
                            "event": "close",
                            **asdict(trade),
                            **asdict(settlement),
                        },
                    )

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
                    **settlement_payload,
                }
                append_jsonl(outcomes_path, result_payload)
                logger.info("paper_resolution", extra={"extra": result_payload})

            summary = {
                "ts_utc": now.isoformat(),
                "mode": "PAPER_5M",
                "runtime_minutes": round((now - started_at).total_seconds() / 60.0, 3),
                "predictions_total": total,
                "predictions_correct": correct,
                "accuracy": (correct / total) if total > 0 else 0.0,
                "pending": [asdict(x) for x in pending.values()],
                "pending_predictions": [asdict(x) for x in pending.values()],
                "deferred_timing": [asdict(x) for x in deferred_timing.values()],
                "pending_trades": [asdict(x) for x in pending_trades.values()],
                "pending_prediction_count": len(pending),
                "deferred_timing_count": len(deferred_timing),
                "pending_trade_count": count_active_pending_paper_trades(pending_trades),
                "resolved_count": len(resolved_ids),
                "risk_state": asdict(risk.state),
                "paper_bankroll": {
                    **asdict(paper_bankroll),
                    "equity": float(paper_bankroll.available_cash + paper_bankroll.reserved_cash),
                },
            }
            write_json(state_path, summary)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logger.info("paper_stopped", extra={"extra": {"reason": "keyboard_interrupt"}})

    end_now = datetime.now(UTC)
    summary = {
        "ts_utc": end_now.isoformat(),
        "mode": "PAPER_5M",
        "runtime_minutes": round((end_now - started_at).total_seconds() / 60.0, 3),
        "predictions_total": total,
        "predictions_correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "pending": [asdict(x) for x in pending.values()],
        "pending_predictions": [asdict(x) for x in pending.values()],
        "deferred_timing": [asdict(x) for x in deferred_timing.values()],
        "pending_trades": [asdict(x) for x in pending_trades.values()],
        "pending_prediction_count": len(pending),
        "deferred_timing_count": len(deferred_timing),
        "pending_trade_count": count_active_pending_paper_trades(pending_trades),
        "resolved_count": len(resolved_ids),
        "risk_state": asdict(risk.state),
        "paper_bankroll": {
            **asdict(paper_bankroll),
            "equity": float(paper_bankroll.available_cash + paper_bankroll.reserved_cash),
        },
    }
    write_json(state_path, summary)

    day_path = Path(cfg["paths"]["outputs_dir"]) / f"paper_5m_summary_{end_now.strftime('%Y%m%d')}.json"
    write_json(day_path, summary)
    return summary


