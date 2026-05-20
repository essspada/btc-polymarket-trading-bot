"""One-shot sim/live entrypoint: discover markets, decide once per window.

Unlike monitor/paper, this mode performs a single pass over the discovered
window markets and exits. It also reconciles open live orders against the
remote exchange and (when live_trading is enabled) caps exposure against the
on-chain collateral snapshot.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.polymarket.book_quality import BookQualityConfig, assess_market_books
from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.clients.gamma_client import GammaClient
from src.polymarket.execution import (
    ExecutionEngine,
    apply_reconcile_result,
    build_live_order_record,
    load_live_order_records,
    save_live_order_records,
)
from src.polymarket.fees import FeeModelConfig
from src.polymarket.market_discovery import discover_candidate_markets
from src.polymarket.risk import RiskManager
from src.runtime.persistence import write_json
from src.runtime.pricing import (
    actionability_calibration_cfg,
    actionability_context,
    skip_confidence_band,
)
from src.runtime.sizing import (
    apply_sizing_cap_to_intent,
    apply_stateful_risk_multiplier_to_intent,
    cash_required_for_order,
)
from src.runtime.snapshots import build_market_books
from src.runtime.timing import risk_window_exposure_multiplier
from src.runtime.utilities import (
    fetch_taker_fee_bps_by_token,
    maker_fill_probability_by_token,
    resolve_output_path,
    to_optional_float,
)
from src.strategy.model_wrapper import MarkovProxyModel, ModelContext
from src.strategy.signals import decide_trade
from src.strategy.sizing import cap_exposure_by_balance
from src.utils.logging import build_logger


def run_sim(cfg: dict[str, Any], confirm_live: bool = False) -> dict[str, Any]:
    now = datetime.now(UTC)
    logger = build_logger(cfg["app"]["name"], cfg["paths"]["logs_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])

    gamma = GammaClient(cfg["polymarket"]["gamma_base_url"])
    clob = ClobClient(cfg["polymarket"]["clob_base_url"])

    markets = discover_candidate_markets(gamma, cfg)
    logger.info("discovered_markets", extra={"extra": {"count": len(markets)}})

    risk_cfg = cfg["risk"]
    exec_cfg = cfg["execution"]
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

    fee_cfg = FeeModelConfig(**cfg["fees"])
    allowed_order_types = exec_cfg.get("allowed_order_types", ["maker", "taker"])
    book_quality_cfg = BookQualityConfig.from_dict(cfg.get("book_quality", {}))
    model = MarkovProxyModel(default_confidence=float(cfg["model"].get("default_confidence", 0.5)))
    executor = ExecutionEngine(
        live_trading=bool(cfg["execution"]["live_trading"]),
        require_confirm_live=bool(cfg["execution"].get("require_confirm_live", True)),
        clob_client=clob,
    )
    live_order_state_path = resolve_output_path(
        outputs_dir,
        str(cfg["execution"].get("live_order_state_file", "live_order_state.json")),
    )
    live_orders = (
        load_live_order_records(live_order_state_path)
        if bool(cfg["execution"]["live_trading"])
        else {}
    )

    trades: list[dict[str, Any]] = []
    latency_buffer_seconds = max(0.0, float(cfg["execution"].get("latency_buffer_seconds", 0)))
    live_wallet_snapshot: dict[str, Any] | None = None

    for order_id, record in list(live_orders.items()):
        if record.terminal:
            continue
        try:
            reconcile = executor.reconcile_order(order_id, now=now, confirm_live=confirm_live)
            apply_reconcile_result(record, reconcile, now)
            logger.info(
                "live_order_reconciled",
                extra={
                    "extra": {
                        "order_id": order_id,
                        "status": reconcile.status,
                        "terminal": reconcile.terminal,
                        "fill_size": reconcile.fill_size,
                        "fill_price": reconcile.fill_price,
                    }
                },
            )
        except Exception as exc:
            logger.info(
                "live_order_reconcile_error",
                extra={"extra": {"order_id": order_id, "error": str(exc)}},
            )

    for m in markets[: int(cfg["polymarket"].get("lookahead_windows", 12))]:
        books, book_error = build_market_books(clob, m)
        if books is None:
            logger.info(
                "skip_market_no_book",
                extra={"extra": {"market": m.market_slug, "market_id": m.market_id, "error": book_error}},
            )
            continue

        quality = assess_market_books(books, book_quality_cfg)
        if not quality.ok:
            logger.info(
                "skip_market_book_quality",
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

        sec_to_exp = max(0.0, (m.end_time - now).total_seconds())
        if sec_to_exp <= latency_buffer_seconds:
            logger.info(
                "skip_market_latency_buffer",
                extra={
                    "extra": {
                        "market": m.market_slug,
                        "seconds_to_expiry": sec_to_exp,
                        "latency_buffer_seconds": latency_buffer_seconds,
                    }
                },
            )
            continue

        fee_bps_map = fetch_taker_fee_bps_by_token(clob=clob, books=books)
        p_up = model.predict_proba(
            ModelContext(
                up_mid=books.up.midpoint,
                down_mid=books.down.midpoint,
                up_imbalance=books.up.topk_imbalance,
                down_imbalance=books.down.topk_imbalance,
                seconds_to_expiry=sec_to_exp,
            )
        )

        max_exposure_usd = float(cfg["risk"]["max_exposure_per_window_usd"])
        if bool(cfg["execution"]["live_trading"]):
            try:
                live_wallet_snapshot = clob.get_collateral_balance_allowance(refresh=True)
            except Exception as exc:
                logger.info(
                    "skip_market_live_wallet_balance_error",
                    extra={"extra": {"market": m.market_slug, "market_id": m.market_id, "error": str(exc)}},
                )
                continue
            max_exposure_usd = cap_exposure_by_balance(
                max_exposure_usd=max_exposure_usd,
                balance_usd=float(live_wallet_snapshot.get("available_to_trade", 0.0)),
                max_balance_fraction_per_trade=float(cfg["risk"].get("max_balance_fraction_per_trade", 1.0)),
            )
            if max_exposure_usd <= 0.0:
                logger.info(
                    "skip_market_live_wallet_balance_zero",
                    extra={
                        "extra": {
                            "market": m.market_slug,
                            "market_id": m.market_id,
                            "available_to_trade": live_wallet_snapshot.get("available_to_trade"),
                            "balance": live_wallet_snapshot.get("balance"),
                            "allowance": live_wallet_snapshot.get("allowance"),
                        }
                    },
                )
                continue

        max_exposure_usd *= risk_window_exposure_multiplier(now=now, risk_cfg=cfg["risk"])

        decision = decide_trade(
            market_slug=m.market_slug,
            books=books,
            p_up=p_up,
            min_edge_to_trade=float(cfg["execution"]["min_edge_to_trade"]),
            min_edge_for_taker=float(cfg["execution"]["min_edge_for_taker"]),
            max_exposure_usd=float(max_exposure_usd),
            maker_preference=bool(cfg["execution"].get("maker_preference", True)),
            fee_cfg=fee_cfg,
            taker_slippage_bps=float(cfg["execution"]["slippage_bps_taker"]),
            maker_slippage_bps=float(cfg["execution"]["slippage_bps_maker"]),
            taker_fee_bps_by_token=fee_bps_map,
            maker_fill_probability=float(cfg["execution"].get("maker_fill_probability", 0.65)),
            maker_fill_probability_by_token=maker_fill_probability_by_token(
                books=books,
                seconds_to_expiry=sec_to_exp,
                exec_cfg=cfg["execution"],
            ),
            maker_ev_advantage_required=float(cfg["execution"].get("maker_ev_advantage_required", 0.0005)),
            allowed_order_types=list(allowed_order_types) if isinstance(allowed_order_types, list) else allowed_order_types,
            lock_side_to_prediction=bool(cfg["execution"].get("lock_side_to_prediction", True)),
            min_reward_to_risk_ratio=float(cfg["execution"].get("min_reward_to_risk_ratio", 0.0)),
            min_expected_roi_cash=to_optional_float(cfg["execution"].get("min_expected_roi_cash")),
            min_breakeven_margin=to_optional_float(cfg["execution"].get("min_breakeven_margin")),
            sizing_mode=str(cfg["execution"].get("sizing_mode", "edge_scaled")),
            max_entry_price=to_optional_float(cfg["execution"].get("max_entry_price")),
            actionability_calibration=actionability_calibration_cfg(cfg["execution"]),
            actionability_context=actionability_context(
                p_up=float(p_up),
                proxy_p_up=None,
                spot_recent_vol_5m_bps=None,
            ),
            skip_confidence_band=skip_confidence_band(cfg["execution"]),
        )

        if decision.intent is None:
            logger.info(
                "no_trade",
                extra={"extra": {"market": m.market_slug, "reason": decision.reason, "p_up": decision.p_up}},
            )
            continue

        selected_taker_fee_bps = fee_bps_map.get(decision.intent.token_id)
        raw_cash_required = cash_required_for_order(
            price=float(decision.intent.price),
            size=float(decision.intent.size),
            order_type=str(decision.intent.order_type),
            fee_cfg=fee_cfg,
            taker_slippage_bps=float(cfg["execution"]["slippage_bps_taker"]),
            maker_slippage_bps=float(cfg["execution"]["slippage_bps_maker"]),
            taker_fee_bps=selected_taker_fee_bps,
        )
        stateful_multiplier, stateful_reason = risk.stateful_trade_multiplier(
            available_cash=(
                float(live_wallet_snapshot.get("available_to_trade", 0.0))
                if isinstance(live_wallet_snapshot, dict)
                else None
            ),
            peak_cash=(
                float(live_wallet_snapshot.get("balance", 0.0))
                if isinstance(live_wallet_snapshot, dict)
                else None
            ),
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
            base_exposure_usd=float(max_exposure_usd),
            risk_cfg=cfg["risk"],
        )

        ok, why = risk.can_trade(now, exposure_usd=float(cash_required))
        if not ok:
            logger.info(
                "risk_block",
                extra={
                    "extra": {
                        "market": m.market_slug,
                        "reason": why,
                        "raw_cash_required": float(raw_cash_required),
                        "cash_required": float(cash_required),
                        **stateful_payload,
                        **sizing_cap_payload,
                    }
                },
            )
            continue

        risk.on_trade_open()
        result = executor.execute(decision.intent, now=now, confirm_live=confirm_live)
        live_order_record = build_live_order_record(
            market_id=m.market_id,
            intent=decision.intent,
            result=result,
            now=now,
        )
        if live_order_record is not None:
            live_orders[live_order_record.order_id] = live_order_record

        trade_row = {
            "market_slug": m.market_slug,
            "market_id": m.market_id,
            "series_slug": m.series_slug,
            "resolution_source": m.resolution_source,
            "seconds_to_expiry": sec_to_exp,
            "p_up": decision.p_up,
            "p_down": decision.p_down,
            "edge": decision.best_edge,
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
            "intent": asdict(decision.intent),
            "execution": asdict(result),
            "live_order_record": (asdict(live_order_record) if live_order_record is not None else None),
            "book_quality": asdict(quality),
            "taker_fee_bps_by_token": fee_bps_map,
            "effective_max_exposure_usd": float(max_exposure_usd),
            "raw_cash_required": float(raw_cash_required),
            **stateful_payload,
            "cash_required": float(cash_required),
            "live_wallet_snapshot": (dict(live_wallet_snapshot) if live_wallet_snapshot is not None else None),
            **sizing_cap_payload,
        }
        trades.append(trade_row)
        logger.info("trade_decision", extra={"extra": trade_row})

    summary = {
        "ts_utc": now.isoformat(),
        "mode": "SIM" if not cfg["execution"]["live_trading"] else "LIVE",
        "discovered": len(markets),
        "trades_considered": len(trades),
        "trades": trades,
        "live_order_state_file": str(live_order_state_path),
        "live_order_count": len(live_orders),
        "live_open_order_count": sum(1 for record in live_orders.values() if not record.terminal),
        "live_terminal_order_count": sum(1 for record in live_orders.values() if record.terminal),
        "risk_state": asdict(risk.state),
        "live_wallet_snapshot": (dict(live_wallet_snapshot) if live_wallet_snapshot is not None else None),
    }

    if bool(cfg["execution"]["live_trading"]):
        save_live_order_records(live_order_state_path, live_orders, now)

    out_path = outputs_dir / f"daily_summary_{now.strftime('%Y%m%d')}.json"
    write_json(out_path, summary)
    return summary
