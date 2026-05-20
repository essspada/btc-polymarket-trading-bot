from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from src.execution.paper_execution import PaperExecutionEngine
from src.polymarket.execution import OrderIntent
from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.polymarket.risk import RiskManager
from src.strategy.confirmation_gate import apply_confirmation_gate
from src.strategy.signals import decide_trade
from src.strategy.sizing import cap_exposure_by_balance, size_from_edge


@dataclass
class LiveLikeConfig:
    initial_balance: float = 100.0
    min_trade_notional: float = 5.0
    require_orderbook: bool = False
    include_trade_log: bool = False
    use_normalized_entry_prices: bool = True
    max_raw_entry_sum_deviation: float = 0.02


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _valid_price(value: Any) -> bool:
    value_f = _to_float(value)
    return value_f is not None and 0.0 < value_f < 1.0


def _candidate_prob(row: dict[str, Any], candidate_key: str) -> float | None:
    aliases = {
        "market_price_baseline": "market_p_up",
        "market_p_up": "market_p_up",
        "spot_window_path": "spot_window_path_p_up",
        "spot_window_path_p_up": "spot_window_path_p_up",
        "spot_market_blend": "spot_market_blend_p_up",
        "spot_market_blend_p_up": "spot_market_blend_p_up",
        "spot_logistic_online": "spot_logistic_online_p_up",
        "spot_logistic_online_p_up": "spot_logistic_online_p_up",
        "spot_consensus_blend": "spot_consensus_blend_p_up",
        "spot_consensus_blend_p_up": "spot_consensus_blend_p_up",
        "spot_timing_policy": "spot_timing_policy_p_up",
        "spot_timing_policy_p_up": "spot_timing_policy_p_up",
    }
    key = aliases.get(str(candidate_key).strip(), str(candidate_key).strip())
    value = _to_float(row.get(key))
    if value is None:
        return None
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _candidate_prefix(candidate_key: str) -> str:
    aliases = {
        "market_price_baseline": "market_p_up",
        "market_p_up": "market_p_up",
        "spot_window_path": "spot_window_path_p_up",
        "spot_window_path_p_up": "spot_window_path_p_up",
        "spot_market_blend": "spot_market_blend_p_up",
        "spot_market_blend_p_up": "spot_market_blend_p_up",
        "spot_logistic_online": "spot_logistic_online_p_up",
        "spot_logistic_online_p_up": "spot_logistic_online_p_up",
        "spot_consensus_blend": "spot_consensus_blend_p_up",
        "spot_consensus_blend_p_up": "spot_consensus_blend_p_up",
        "spot_timing_policy": "spot_timing_policy_p_up",
        "spot_timing_policy_p_up": "spot_timing_policy_p_up",
    }
    key = aliases.get(str(candidate_key).strip(), str(candidate_key).strip())
    return key[:-5] if key.endswith("_p_up") else key


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _sort_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(row: dict[str, Any]) -> str:
        for field in ("created_at", "ts_utc", "end_time"):
            if row.get(field):
                return str(row.get(field))
        return ""

    return sorted(list(rows), key=_key)


def _date_key(row: dict[str, Any]) -> str:
    return str(row.get("created_at") or row.get("ts_utc") or "")[:10]


def _build_books_from_row(row: dict[str, Any]) -> MarketBooks | None:
    up_ask = _to_float(row.get("up_best_ask"))
    down_ask = _to_float(row.get("down_best_ask"))
    if not (_valid_price(up_ask) and _valid_price(down_ask)):
        return None
    up_bid = _to_float(row.get("up_best_bid"), 0.0) or 0.0
    down_bid = _to_float(row.get("down_best_bid"), 0.0) or 0.0
    up_mid = _to_float(row.get("up_midpoint"), (up_bid + up_ask) / 2.0)
    down_mid = _to_float(row.get("down_midpoint"), (down_bid + down_ask) / 2.0)
    up_spread = _to_float(row.get("up_spread"), max(0.0, up_ask - up_bid)) or max(0.0, up_ask - up_bid)
    down_spread = _to_float(row.get("down_spread"), max(0.0, down_ask - down_bid)) or max(0.0, down_ask - down_bid)
    return MarketBooks(
        up=OutcomeBook(
            token_id=str(row.get("up_token_id") or "UP"),
            best_bid=float(up_bid),
            best_ask=float(up_ask),
            midpoint=float(up_mid),
            spread=float(up_spread),
            topk_imbalance=float(_to_float(row.get("up_topk_imbalance"), 0.0) or 0.0),
            best_bid_size=float(_to_float(row.get("up_best_bid_size"), 0.0) or 0.0),
            best_ask_size=float(_to_float(row.get("up_best_ask_size"), 0.0) or 0.0),
            top3_bid_size=float(_to_float(row.get("up_top3_bid_size"), 0.0) or 0.0),
            top3_ask_size=float(_to_float(row.get("up_top3_ask_size"), 0.0) or 0.0),
        ),
        down=OutcomeBook(
            token_id=str(row.get("down_token_id") or "DOWN"),
            best_bid=float(down_bid),
            best_ask=float(down_ask),
            midpoint=float(down_mid),
            spread=float(down_spread),
            topk_imbalance=float(_to_float(row.get("down_topk_imbalance"), 0.0) or 0.0),
            best_bid_size=float(_to_float(row.get("down_best_bid_size"), 0.0) or 0.0),
            best_ask_size=float(_to_float(row.get("down_best_ask_size"), 0.0) or 0.0),
            top3_bid_size=float(_to_float(row.get("down_top3_bid_size"), 0.0) or 0.0),
            top3_ask_size=float(_to_float(row.get("down_top3_ask_size"), 0.0) or 0.0),
        ),
    )


def _taker_fee_bps_for_token(row: dict[str, Any], token_id: str, up_token_id: str, down_token_id: str) -> float | None:
    if token_id == up_token_id:
        return _to_float(row.get("up_taker_fee_bps"))
    if token_id == down_token_id:
        return _to_float(row.get("down_taker_fee_bps"))
    return None


def _trade_fee_cfg(fee_cfg: FeeModelConfig, is_taker: bool, taker_fee_bps: float | None) -> FeeModelConfig:
    if not is_taker or taker_fee_bps is None:
        return fee_cfg
    return FeeModelConfig(
        maker_fee_bps=float(fee_cfg.maker_fee_bps),
        taker_fee_bps=float(taker_fee_bps),
        curve_rate=float(fee_cfg.curve_rate),
        curve_exponent=float(fee_cfg.curve_exponent),
        min_fee=float(fee_cfg.min_fee),
    )


def _cash_required_per_share(
    *,
    price: float,
    is_taker: bool,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    taker_fee_bps: float | None,
) -> float:
    cfg_for_trade = _trade_fee_cfg(fee_cfg, is_taker=is_taker, taker_fee_bps=taker_fee_bps)
    fee_per_share = compute_trade_fee(price=price, size=1.0, is_taker=is_taker, cfg=cfg_for_trade)
    slip_per_share = float(price) * (float(slippage_bps) / 10_000.0)
    return float(price) + float(fee_per_share) + float(slip_per_share)


def _scale_intent_to_balance(
    intent: OrderIntent,
    *,
    balance: float,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    taker_fee_bps: float | None,
) -> tuple[OrderIntent | None, float]:
    if balance <= 0.0:
        return None, 0.0
    is_taker = str(intent.order_type).upper() == "TAKER"
    cash_per_share = _cash_required_per_share(
        price=float(intent.price),
        is_taker=is_taker,
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
        taker_fee_bps=taker_fee_bps,
    )
    if cash_per_share <= 0.0:
        return None, 0.0
    max_size = float(balance) / float(cash_per_share)
    size = min(float(intent.size), float(max_size))
    if size <= 0.0:
        return None, 0.0
    scaled_intent = OrderIntent(
        market_slug=str(intent.market_slug),
        token_id=str(intent.token_id),
        side=str(intent.side),
        order_type=str(intent.order_type),
        price=float(intent.price),
        size=float(size),
        expected_edge=float(intent.expected_edge),
    )
    return scaled_intent, float(size * cash_per_share)


def _apply_stateful_multiplier_to_intent(
    intent: OrderIntent,
    *,
    cash_required: float,
    multiplier: float,
) -> tuple[OrderIntent | None, float, dict[str, Any]]:
    multiplier_f = max(0.0, min(1.0, float(multiplier)))
    payload = {
        "stateful_multiplier": float(multiplier_f),
        "stateful_multiplier_applied": bool(multiplier_f < 0.999),
        "stateful_raw_cash_required": float(max(0.0, cash_required)),
        "stateful_capped_cash_required": float(max(0.0, cash_required) * multiplier_f),
    }
    if multiplier_f <= 0.0:
        return None, 0.0, payload
    if multiplier_f >= 0.999:
        return intent, float(cash_required), payload
    return replace(intent, size=float(intent.size) * multiplier_f), float(cash_required) * multiplier_f, payload


def _maker_fill_probability_map(books: MarketBooks, seconds_to_expiry: float, exec_cfg: dict[str, Any]) -> dict[str, float]:
    engine = PaperExecutionEngine(
        maker_fill_floor=float(exec_cfg.get("maker_fill_floor", 0.05)),
        maker_fill_cap=float(exec_cfg.get("maker_fill_cap", 0.9)),
        maker_fill_base=float(exec_cfg.get("maker_fill_base", 0.78)),
        maker_fill_spread_penalty=float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        maker_fill_late_penalty_90=float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        maker_fill_late_penalty_45=float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    )
    return {
        books.up.token_id: engine._maker_fill_probability(float(books.up.spread), float(seconds_to_expiry)),
        books.down.token_id: engine._maker_fill_probability(float(books.down.spread), float(seconds_to_expiry)),
    }


def _legacy_trade_intent(
    row: dict[str, Any],
    *,
    candidate_key: str,
    balance: float,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    live_like_cfg: LiveLikeConfig,
) -> tuple[OrderIntent | None, float | None, str]:
    p_up = _candidate_prob(row, candidate_key)
    if p_up is None:
        return None, None, "missing_probability"

    prefix = _candidate_prefix(candidate_key)
    selected_side = str(row.get(f"{prefix}_selected_side") or "").strip().lower()
    if selected_side not in {"up", "down"}:
        selected_side = "up" if float(p_up) >= 0.5 else "down"

    raw_sum = _to_float(row.get("entry_price_sum_raw"))
    if raw_sum is not None and abs(float(raw_sum) - 1.0) > float(live_like_cfg.max_raw_entry_sum_deviation):
        return None, None, "entry_price_sum_out_of_bounds"

    up_price = _to_float(row.get("up_entry_price_norm")) if live_like_cfg.use_normalized_entry_prices else None
    down_price = _to_float(row.get("down_entry_price_norm")) if live_like_cfg.use_normalized_entry_prices else None
    if not (_valid_price(up_price) and _valid_price(down_price)):
        up_price = _to_float(row.get("up_entry_price"))
        down_price = _to_float(row.get("down_entry_price"))
    if not (_valid_price(up_price) and _valid_price(down_price)):
        return None, None, "missing_entry_price"

    price = float(up_price if selected_side == "up" else down_price)
    p_side = float(p_up if selected_side == "up" else 1.0 - p_up)
    raw_edge = float(p_side - price)
    max_exposure_usd = cap_exposure_by_balance(
        max_exposure_usd=float(risk_cfg.get("max_exposure_per_window_usd", balance)),
        balance_usd=float(balance),
        max_balance_fraction_per_trade=float(risk_cfg.get("max_balance_fraction_per_trade", 1.0)),
    )
    usd_size = size_from_edge(
        max_exposure_usd=max_exposure_usd,
        edge=raw_edge,
        min_edge=float(exec_cfg.get("min_edge_to_trade", 0.004)),
    )
    if usd_size <= 0.0:
        return None, None, "raw_edge_below_threshold"

    ref_price = max(float(up_price), float(down_price), 0.01)
    size_shares = float(usd_size / ref_price)
    order_type = "TAKER"
    cash_per_share = _cash_required_per_share(
        price=price,
        is_taker=True,
        fee_cfg=fee_cfg,
        slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
        taker_fee_bps=None,
    )
    edge_per_share = float(p_side - cash_per_share)
    if edge_per_share < float(exec_cfg.get("min_edge_to_trade", 0.004)):
        return None, None, "net_edge_below_threshold"
    if (size_shares * cash_per_share) < float(exec_cfg.get("min_trade_notional", 0.0) or 0.0):
        return None, None, "trade_notional_too_small"

    token_id = str(row.get("up_token_id") if selected_side == "up" else row.get("down_token_id"))
    return (
        OrderIntent(
            market_slug=str(row.get("market_slug") or row.get("market_id") or "legacy_market"),
            token_id=token_id,
            side="BUY",
            order_type=order_type,
            price=price,
            size=float(size_shares),
            expected_edge=float(edge_per_share),
        ),
        None,
        "ok",
    )


def run_live_like_backtest(
    rows: Sequence[dict[str, Any]],
    *,
    candidate_key: str,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    config: LiveLikeConfig,
) -> dict[str, Any]:
    ordered = _sort_rows(rows)
    balance = float(config.initial_balance)
    peak_balance = float(balance)
    max_drawdown_pct = 0.0
    warnings: list[str] = []

    risk = RiskManager(
        max_exposure_per_window_usd=float(risk_cfg.get("max_exposure_per_window_usd", balance)),
        max_daily_loss_usd=float(risk_cfg.get("max_daily_loss_usd", max(balance, 1.0))),
        cooldown_after_loss_streak=int(risk_cfg.get("cooldown_after_loss_streak", 999_999)),
        cooldown_windows=int(risk_cfg.get("cooldown_windows", 0)),
        max_open_positions=int(risk_cfg.get("max_open_positions", 1)),
        drift_enabled=bool(risk_cfg.get("drift_enabled", False)),
        drift_lookback=int(risk_cfg.get("drift_lookback", 60)),
        drift_min_samples=int(risk_cfg.get("drift_min_samples", 40)),
        drift_min_accuracy=float(risk_cfg.get("drift_min_accuracy", 0.46)),
        drift_cooldown_windows=int(risk_cfg.get("drift_cooldown_windows", 6)),
        stateful_regime=risk_cfg.get("stateful_regime", {}),
    )
    paper = PaperExecutionEngine(
        maker_fill_floor=float(exec_cfg.get("maker_fill_floor", 0.05)),
        maker_fill_cap=float(exec_cfg.get("maker_fill_cap", 0.9)),
        maker_fill_base=float(exec_cfg.get("maker_fill_base", 0.78)),
        maker_fill_spread_penalty=float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        maker_fill_late_penalty_90=float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        maker_fill_late_penalty_45=float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    )

    current_day = ""
    daily: dict[str, dict[str, Any]] = {}
    skip_reason_counts: Counter[str] = Counter()
    data_mode_counts: Counter[str] = Counter()
    order_type_counts: Counter[str] = Counter()
    stateful_reason_counts: Counter[str] = Counter()
    trade_logs: list[dict[str, Any]] = []

    def _ensure_day(day: str) -> dict[str, Any]:
        nonlocal current_day
        if day != current_day:
            current_day = day
            daily.setdefault(
                day,
                {
                    "start_balance": float(balance),
                    "end_balance": float(balance),
                    "pnl": 0.0,
                    "submitted_trades": 0,
                    "filled_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "skips": Counter(),
                    "best_trade_pnl": 0.0,
                    "worst_trade_pnl": 0.0,
                },
            )
        return daily[day]

    for idx, row in enumerate(ordered, start=1):
        day = _date_key(row)
        daily_entry = _ensure_day(day)
        created_at = _parse_ts(row.get("created_at") or row.get("ts_utc"))
        if created_at is None:
            skip_reason_counts["missing_timestamp"] += 1
            daily_entry["skips"]["missing_timestamp"] += 1
            continue

        books = _build_books_from_row(row)
        native_mode = books is not None
        if not native_mode and bool(config.require_orderbook):
            skip_reason_counts["orderbook_required"] += 1
            daily_entry["skips"]["orderbook_required"] += 1
            continue

        if native_mode:
            p_up = _candidate_prob(row, candidate_key)
            if p_up is None:
                skip_reason_counts["missing_probability"] += 1
                daily_entry["skips"]["missing_probability"] += 1
                continue
            seconds_to_expiry = float(_to_float(row.get("seconds_to_expiry"), 300.0) or 300.0)
            capped_exposure = cap_exposure_by_balance(
                max_exposure_usd=float(risk_cfg.get("max_exposure_per_window_usd", balance)),
                balance_usd=float(balance),
                max_balance_fraction_per_trade=float(risk_cfg.get("max_balance_fraction_per_trade", 1.0)),
            )
            decision = decide_trade(
                market_slug=str(row.get("market_slug") or row.get("market_id") or f"row_{idx}"),
                books=books,
                p_up=float(p_up),
                min_edge_to_trade=float(exec_cfg.get("min_edge_to_trade", 0.004)),
                min_edge_for_taker=float(exec_cfg.get("min_edge_for_taker", 0.009)),
                max_exposure_usd=float(capped_exposure),
                maker_preference=bool(exec_cfg.get("maker_preference", True)),
                fee_cfg=fee_cfg,
                taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
                maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3.0)),
                taker_fee_bps_by_token={
                    books.up.token_id: float(_to_float(row.get("up_taker_fee_bps"), fee_cfg.taker_fee_bps) or fee_cfg.taker_fee_bps),
                    books.down.token_id: float(_to_float(row.get("down_taker_fee_bps"), fee_cfg.taker_fee_bps) or fee_cfg.taker_fee_bps),
                },
                maker_fill_probability=float(exec_cfg.get("maker_fill_probability", 0.65)),
                maker_fill_probability_by_token=_maker_fill_probability_map(books, seconds_to_expiry, exec_cfg),
                maker_ev_advantage_required=float(exec_cfg.get("maker_ev_advantage_required", 0.0005)),
                allowed_order_types=list(exec_cfg.get("allowed_order_types", ["maker", "taker"])),
                lock_side_to_prediction=bool(exec_cfg.get("lock_side_to_prediction", True)),
                min_reward_to_risk_ratio=float(exec_cfg.get("min_reward_to_risk_ratio", 0.0)),
                sizing_mode=str(exec_cfg.get("sizing_mode", "edge_scaled")),
            )
            decision, _confirmation_gate_payload = apply_confirmation_gate(
                decision=decision,
                books=books,
                candidate_models=row.get("candidate_models") if isinstance(row.get("candidate_models"), dict) else {},
                gate_cfg=exec_cfg.get("confirmation_gate", {}),
            )
            if decision.intent is None:
                reason = str(decision.reason or "no_trade")
                skip_reason_counts[reason] += 1
                daily_entry["skips"][reason] += 1
                continue
            taker_fee_bps = _taker_fee_bps_for_token(
                row,
                token_id=decision.intent.token_id,
                up_token_id=books.up.token_id,
                down_token_id=books.down.token_id,
            )
            intent, cash_required = _scale_intent_to_balance(
                decision.intent,
                balance=float(balance),
                fee_cfg=fee_cfg,
                slippage_bps=float(
                    exec_cfg.get("slippage_bps_taker", 12.0)
                    if str(decision.intent.order_type).upper() == "TAKER"
                    else exec_cfg.get("slippage_bps_maker", 3.0)
                ),
                taker_fee_bps=taker_fee_bps,
            )
            data_mode = "native_orderbook_runtime"
            if intent is None:
                skip_reason_counts["insufficient_balance"] += 1
                daily_entry["skips"]["insufficient_balance"] += 1
                continue
            spread = float(books.up.spread if intent.token_id == books.up.token_id else books.down.spread)
        else:
            if "legacy_entry_replay" not in warnings:
                warnings.append("legacy_entry_replay")
            intent, _, reason = _legacy_trade_intent(
                row,
                candidate_key=candidate_key,
                balance=float(balance),
                fee_cfg=fee_cfg,
                exec_cfg=exec_cfg,
                risk_cfg=risk_cfg,
                live_like_cfg=config,
            )
            if intent is None:
                skip_reason_counts[reason] += 1
                daily_entry["skips"][reason] += 1
                continue
            intent, cash_required = _scale_intent_to_balance(
                intent,
                balance=float(balance),
                fee_cfg=fee_cfg,
                slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
                taker_fee_bps=None,
            )
            data_mode = "legacy_entry_replay"
            if intent is None:
                skip_reason_counts["insufficient_balance"] += 1
                daily_entry["skips"]["insufficient_balance"] += 1
                continue
            seconds_to_expiry = float(_to_float(row.get("seconds_to_expiry"), 300.0) or 300.0)
            spread = 0.0
            up_token_id = str(row.get("up_token_id") or "UP")
            down_token_id = str(row.get("down_token_id") or "DOWN")

        stateful_multiplier, stateful_reason = risk.stateful_trade_multiplier(
            available_cash=float(balance),
            peak_cash=float(peak_balance),
            ts=created_at,
        )
        intent, cash_required, stateful_payload = _apply_stateful_multiplier_to_intent(
            intent,
            cash_required=float(cash_required),
            multiplier=float(stateful_multiplier),
        )
        stateful_payload["stateful_reason"] = str(stateful_reason)
        stateful_reason_counts[str(stateful_reason)] += 1
        if intent is None:
            skip_reason_counts["stateful_non_positive_multiplier"] += 1
            daily_entry["skips"]["stateful_non_positive_multiplier"] += 1
            continue

        if cash_required < float(config.min_trade_notional):
            skip_reason_counts["trade_notional_too_small"] += 1
            daily_entry["skips"]["trade_notional_too_small"] += 1
            continue

        ok, risk_reason = risk.can_trade(created_at, exposure_usd=float(cash_required))
        if not ok:
            skip_reason_counts[str(risk_reason)] += 1
            daily_entry["skips"][str(risk_reason)] += 1
            continue

        order_type = str(intent.order_type).upper()
        up_token_id = str(row.get("up_token_id") or "UP")
        down_token_id = str(row.get("down_token_id") or "DOWN")
        selected_taker_fee_bps = _taker_fee_bps_for_token(row, intent.token_id, up_token_id, down_token_id)

        risk.on_trade_open()
        trade = paper.open_trade(
            intent=intent,
            market_id=str(row.get("market_id") or f"market_{idx}"),
            created_at=created_at,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            spread=float(spread),
            seconds_to_expiry=float(seconds_to_expiry),
            taker_fee_bps=selected_taker_fee_bps,
        )
        settled_at = _parse_ts(row.get("resolved_ts_utc") or row.get("end_time") or row.get("created_at")) or created_at
        settlement = paper.settle_trade(
            trade=trade,
            settled_at=settled_at,
            actual_side=str(row.get("actual_side") or "").strip().lower(),
            fee_cfg=fee_cfg,
            taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
            maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3.0)),
        )
        balance += float(settlement.net_pnl)
        peak_balance = max(float(peak_balance), float(balance))
        if peak_balance > 0.0:
            max_drawdown_pct = max(max_drawdown_pct, max(0.0, (peak_balance - balance) / peak_balance))
        risk.on_trade_close(float(settlement.net_pnl))
        if trade.filled:
            chosen_side = "up" if trade.token_id == up_token_id else "down"
            risk.on_outcome(chosen_side == str(row.get("actual_side") or "").strip().lower())

        data_mode_counts[data_mode] += 1
        order_type_counts[order_type] += 1
        daily_entry["submitted_trades"] += 1
        daily_entry["filled_trades"] += int(trade.filled)
        daily_entry["pnl"] += float(settlement.net_pnl)
        daily_entry["end_balance"] = float(balance)
        if settlement.net_pnl > 0.0:
            daily_entry["wins"] += 1
        elif settlement.net_pnl < 0.0:
            daily_entry["losses"] += 1
        daily_entry["best_trade_pnl"] = max(float(daily_entry["best_trade_pnl"]), float(settlement.net_pnl))
        daily_entry["worst_trade_pnl"] = min(float(daily_entry["worst_trade_pnl"]), float(settlement.net_pnl))

        trade_logs.append(
            {
                "trade_index": idx,
                "market_id": row.get("market_id"),
                "created_at": created_at.isoformat(),
                "date": day,
                "data_mode": data_mode,
                "order_type": order_type,
                "fill_price": float(trade.fill_price),
                "fill_size": float(trade.fill_size),
                "filled": bool(trade.filled),
                "fill_probability": float(trade.fill_probability),
                "expected_edge": float(intent.expected_edge),
                "cash_required": float(cash_required),
                **stateful_payload,
                "balance_after": float(balance),
                "actual_side": str(row.get("actual_side") or "").strip().lower(),
                "net_pnl": float(settlement.net_pnl),
                "fee": float(settlement.fee),
                "slippage": float(settlement.slippage),
            }
        )

    ordered_daily = []
    for day, entry in sorted(daily.items()):
        entry["skips"] = dict(entry["skips"])
        entry["win_rate"] = float(entry["wins"] / entry["filled_trades"]) if entry["filled_trades"] > 0 else 0.0
        ordered_daily.append({"date": day, **entry})

    submitted_trades = int(sum(entry["submitted_trades"] for entry in daily.values()))
    filled_trades = int(sum(entry["filled_trades"] for entry in daily.values()))
    wins = int(sum(entry["wins"] for entry in daily.values()))
    losses = int(sum(entry["losses"] for entry in daily.values()))
    positive_days = int(sum(1 for entry in daily.values() if float(entry["pnl"]) > 0.0))
    negative_days = int(sum(1 for entry in daily.values() if float(entry["pnl"]) < 0.0))
    active_days = int(sum(1 for entry in daily.values() if int(entry["submitted_trades"]) > 0))
    worst_day = min(ordered_daily, key=lambda item: item["pnl"]) if ordered_daily else None
    best_day = max(ordered_daily, key=lambda item: item["pnl"]) if ordered_daily else None

    return {
        "candidate_key": str(candidate_key),
        "initial_balance": float(config.initial_balance),
        "final_balance": float(balance),
        "return_multiple": float(balance / config.initial_balance) if config.initial_balance > 0 else 0.0,
        "submitted_trades": submitted_trades,
        "filled_trades": filled_trades,
        "fill_rate": float(filled_trades / submitted_trades) if submitted_trades > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / filled_trades) if filled_trades > 0 else 0.0,
        "max_drawdown_pct": float(max_drawdown_pct),
        "peak_balance": float(peak_balance),
        "active_days": active_days,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "best_day": best_day,
        "worst_day": worst_day,
        "skip_reason_counts": dict(skip_reason_counts),
        "data_mode_counts": dict(data_mode_counts),
        "order_type_counts": dict(order_type_counts),
        "stateful_reason_counts": dict(stateful_reason_counts),
        "warnings": warnings,
        "daily": ordered_daily,
        "trade_log_preview": trade_logs[:20],
        "last_trade": trade_logs[-1] if trade_logs else None,
        "trade_log": trade_logs if config.include_trade_log else None,
        "config": asdict(config),
    }
