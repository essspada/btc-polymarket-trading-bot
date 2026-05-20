from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from src.backtest.live_like import _build_books_from_row, _maker_fill_probability_map, _parse_ts, _to_float
from src.execution.paper_execution import PaperExecutionEngine
from src.polymarket.execution import OrderIntent
from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.polymarket.orderbook import MarketBooks
from src.polymarket.risk import RiskManager
from src.strategy.confirmation_gate import apply_confirmation_gate
from src.strategy.signals import SignalDecision, decide_trade
from src.strategy.sizing import cap_exposure_by_balance


@dataclass
class RedecisionPolicy:
    """A replay policy that re-runs decision logic on saved runtime rows."""

    name: str
    candidate_key: str = "p_up"
    exec_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedecisionConfig:
    initial_balance: float = 100.0
    min_trade_notional: float = 0.0
    assume_filled: bool = True
    fill_mode: str = "assume_filled"  # assume_filled | paper_sim
    include_decisions: bool = False


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_rollout_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load the most useful rows available from a rollout directory.

    Outcomes contain resolved `actual_side`, so they are preferred. Predictions are
    used as a fallback for shadow/decision-only replay.
    """

    base = Path(run_dir)
    for name in ("paper_outcomes_5m.jsonl", "paper_predictions_5m.jsonl"):
        path = base / name
        if path.exists():
            return _merge_trade_token_ids(read_jsonl(path), base / "paper_trades_5m.jsonl")
    raise FileNotFoundError(f"No paper_outcomes_5m.jsonl or paper_predictions_5m.jsonl in {base}")


def _merge_trade_token_ids(rows: list[dict[str, Any]], trades_path: Path) -> list[dict[str, Any]]:
    if not trades_path.exists():
        return rows
    token_map: dict[str, dict[str, str]] = {}
    for trade in read_jsonl(trades_path):
        up_token_id = trade.get("up_token_id")
        down_token_id = trade.get("down_token_id")
        if not up_token_id or not down_token_id:
            continue
        payload = {"up_token_id": str(up_token_id), "down_token_id": str(down_token_id)}
        for key_name in ("market_id", "market_slug"):
            key = trade.get(key_name)
            if key:
                token_map[str(key)] = payload
    if not token_map:
        return rows

    merged: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        lookup = None
        for key_name in ("market_id", "market_slug"):
            key = item.get(key_name)
            if key and str(key) in token_map:
                lookup = token_map[str(key)]
                break
        if lookup is not None:
            item.setdefault("up_token_id", lookup["up_token_id"])
            item.setdefault("down_token_id", lookup["down_token_id"])
        merged.append(item)
    return merged


def _effective_exec_cfg(exec_cfg: dict[str, Any], policy: RedecisionPolicy) -> dict[str, Any]:
    merged = dict(exec_cfg)
    merged.update(policy.exec_overrides)
    return merged


def _candidate_probability(row: dict[str, Any], candidate_key: str) -> float | None:
    key = str(candidate_key).strip()
    direct_keys = [key]
    if not key.endswith("_p_up"):
        direct_keys.append(f"{key}_p_up")
    for direct_key in direct_keys:
        value = _to_float(row.get(direct_key))
        if value is not None:
            return max(1e-6, min(1.0 - 1e-6, float(value)))

    nested = row.get("candidate_models")
    if isinstance(nested, dict):
        item = nested.get(key)
        if isinstance(item, dict):
            for nested_key in ("p_up", "p_up_calibrated", "p_up_pre_calibration", "p_up_raw"):
                value = _to_float(item.get(nested_key))
                if value is not None:
                    return max(1e-6, min(1.0 - 1e-6, float(value)))
    return None


def _fee_cfg_for_intent(fee_cfg: FeeModelConfig, row: dict[str, Any], intent: OrderIntent, books: MarketBooks) -> FeeModelConfig:
    if str(intent.order_type).upper() != "TAKER":
        return fee_cfg
    bps: float | None = None
    if intent.token_id == books.up.token_id:
        bps = _to_float(row.get("up_taker_fee_bps"))
    elif intent.token_id == books.down.token_id:
        bps = _to_float(row.get("down_taker_fee_bps"))
    if bps is None:
        return fee_cfg
    return FeeModelConfig(
        maker_fee_bps=float(fee_cfg.maker_fee_bps),
        taker_fee_bps=float(bps),
        curve_rate=float(fee_cfg.curve_rate),
        curve_exponent=float(fee_cfg.curve_exponent),
        min_fee=float(fee_cfg.min_fee),
    )


def _intent_side(intent: OrderIntent, books: MarketBooks) -> str | None:
    if intent.token_id == books.up.token_id:
        return "up"
    if intent.token_id == books.down.token_id:
        return "down"
    return None


def _apply_runtime_risk_layers(
    *,
    decision: SignalDecision,
    row: dict[str, Any],
    books: MarketBooks,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    base_exposure_usd: float,
    available_cash: float,
    peak_cash: float,
    created_at: Any = None,
    stateful_risk: RiskManager | None = None,
) -> tuple[SignalDecision, dict[str, Any]]:
    if decision.intent is None:
        return decision, {
            "runtime_raw_cash_required": None,
            "runtime_cash_required": decision.cash_required,
            "stateful_payload": {},
            "sizing_cap_payload": {},
            "x3_payload": {},
        }

    # Import lazily so the production runtime remains the single source of truth
    # without creating an import cycle during normal module loading.
    from src.runtime.sizing import (
        apply_sizing_cap_to_intent as _apply_sizing_cap_to_intent,
        apply_stateful_risk_multiplier_to_intent as _apply_stateful_risk_multiplier_to_intent,
    )
    from src.runtime.x3 import apply_x3_active_to_decision as _apply_x3_active_to_decision

    raw_cash_required = _cash_required(decision.intent, row, books, fee_cfg, exec_cfg)
    stateful_payload: dict[str, Any] = {
        "stateful_regime_enabled": bool(risk_cfg.get("stateful_regime", {}).get("enabled", False)),
        "stateful_multiplier_applied": False,
        "stateful_multiplier": 1.0,
        "stateful_reason": "stateful_disabled",
        "stateful_raw_cash_required": float(raw_cash_required),
        "stateful_capped_cash_required": float(raw_cash_required),
    }
    stateful_cash_required = float(raw_cash_required)
    if stateful_risk is not None:
        stateful_multiplier, stateful_reason = stateful_risk.stateful_trade_multiplier(
            available_cash=float(available_cash),
            peak_cash=float(peak_cash),
            ts=_parse_ts(created_at),
        )
        decision, stateful_cash_required, stateful_payload = _apply_stateful_risk_multiplier_to_intent(
            decision=decision,
            raw_cash_required=float(raw_cash_required),
            multiplier=float(stateful_multiplier),
        )
        stateful_payload["stateful_reason"] = str(stateful_reason)
        if decision.intent is None:
            return decision, {
                "runtime_raw_cash_required": float(raw_cash_required),
                "runtime_cash_required": float(stateful_cash_required),
                "stateful_payload": stateful_payload,
                "sizing_cap_payload": {},
                "x3_payload": {},
            }

    updated_intent, cash_required, sizing_payload = _apply_sizing_cap_to_intent(
        intent=decision.intent,
        raw_cash_required=float(stateful_cash_required),
        base_exposure_usd=float(base_exposure_usd),
        risk_cfg=risk_cfg,
    )
    updated_decision = replace(decision, intent=updated_intent, cash_required=float(cash_required))
    selected_side = _intent_side(updated_intent, books)
    updated_decision, cash_required, x3_payload = _apply_x3_active_to_decision(
        risk_cfg=risk_cfg,
        decision=updated_decision,
        candidate_cash_required=float(cash_required),
        raw_cash_required=float(raw_cash_required),
        sizing_cap_payload=sizing_payload,
        candidate_side=selected_side,
    )
    return updated_decision, {
        "runtime_raw_cash_required": float(raw_cash_required),
        "runtime_cash_required": float(cash_required),
        "stateful_payload": stateful_payload,
        "sizing_cap_payload": sizing_payload,
        "x3_payload": x3_payload,
    }


def _cash_required(intent: OrderIntent, row: dict[str, Any], books: MarketBooks, fee_cfg: FeeModelConfig, exec_cfg: dict[str, Any]) -> float:
    is_taker = str(intent.order_type).upper() == "TAKER"
    cfg = _fee_cfg_for_intent(fee_cfg, row, intent, books)
    slippage_bps = float(exec_cfg.get("slippage_bps_taker", 12.0) if is_taker else exec_cfg.get("slippage_bps_maker", 3.0))
    fee = compute_trade_fee(price=float(intent.price), size=float(intent.size), is_taker=is_taker, cfg=cfg)
    slip = float(intent.size) * float(intent.price) * (slippage_bps / 10_000.0)
    return float(float(intent.size) * float(intent.price) + fee + slip)


def _pnl_if_filled(
    intent: OrderIntent,
    row: dict[str, Any],
    books: MarketBooks,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
) -> dict[str, Any]:
    actual_side = str(row.get("actual_side") or "").strip().lower()
    selected_side = _intent_side(intent, books)
    if actual_side not in {"up", "down"} or selected_side not in {"up", "down"}:
        return {
            "actual_side": actual_side or None,
            "selected_side": selected_side,
            "pnl": None,
            "win": None,
            "fee": None,
            "slippage": None,
            "cash_required": _cash_required(intent, row, books, fee_cfg, exec_cfg),
        }

    is_taker = str(intent.order_type).upper() == "TAKER"
    cfg = _fee_cfg_for_intent(fee_cfg, row, intent, books)
    slippage_bps = float(exec_cfg.get("slippage_bps_taker", 12.0) if is_taker else exec_cfg.get("slippage_bps_maker", 3.0))
    payout = 1.0 if selected_side == actual_side else 0.0
    gross = float(intent.size) * (payout - float(intent.price))
    fee = compute_trade_fee(price=float(intent.price), size=float(intent.size), is_taker=is_taker, cfg=cfg)
    slip = float(intent.size) * float(intent.price) * (slippage_bps / 10_000.0)
    pnl = float(gross - fee - slip)
    cash_required = float(float(intent.size) * float(intent.price) + fee + slip)
    return {
        "actual_side": actual_side,
        "selected_side": selected_side,
        "pnl": pnl,
        "win": bool(selected_side == actual_side),
        "fee": float(fee),
        "slippage": float(slip),
        "cash_required": cash_required,
    }


def _paper_simulated_pnl(
    intent: OrderIntent,
    row: dict[str, Any],
    books: MarketBooks,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
) -> dict[str, Any]:
    created_at = _parse_ts(row.get("created_at") or row.get("ts_utc"))
    if created_at is None:
        return {
            "paper_sim_filled": False,
            "paper_sim_pnl": None,
            "paper_sim_fill_probability": None,
            "paper_sim_trade_id": None,
        }
    settled_at = _parse_ts(row.get("resolved_ts_utc") or row.get("end_time") or row.get("created_at")) or created_at
    selected_side = _intent_side(intent, books)
    spread = float(books.up.spread if selected_side == "up" else books.down.spread)
    taker_fee_bps = None
    if intent.token_id == books.up.token_id:
        taker_fee_bps = _to_float(row.get("up_taker_fee_bps"))
    elif intent.token_id == books.down.token_id:
        taker_fee_bps = _to_float(row.get("down_taker_fee_bps"))
    engine = PaperExecutionEngine(
        maker_fill_floor=float(exec_cfg.get("maker_fill_floor", 0.05)),
        maker_fill_cap=float(exec_cfg.get("maker_fill_cap", 0.9)),
        maker_fill_base=float(exec_cfg.get("maker_fill_base", 0.78)),
        maker_fill_spread_penalty=float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        maker_fill_late_penalty_90=float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        maker_fill_late_penalty_45=float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    )
    trade = engine.open_trade(
        intent=intent,
        market_id=str(row.get("market_id") or "redecision_market"),
        created_at=created_at,
        up_token_id=books.up.token_id,
        down_token_id=books.down.token_id,
        spread=spread,
        seconds_to_expiry=float(_to_float(row.get("seconds_to_expiry"), 300.0) or 300.0),
        taker_fee_bps=taker_fee_bps,
    )
    settlement = engine.settle_trade(
        trade=trade,
        settled_at=settled_at,
        actual_side=str(row.get("actual_side") or "").strip().lower(),
        fee_cfg=fee_cfg,
        taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
        maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3.0)),
    )
    return {
        "paper_sim_filled": bool(trade.filled),
        "paper_sim_pnl": float(settlement.net_pnl),
        "paper_sim_fill_probability": float(trade.fill_probability),
        "paper_sim_trade_id": trade.trade_id,
    }


def redecide_row(
    row: dict[str, Any],
    *,
    policy: RedecisionPolicy,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    balance: float,
    peak_balance: float | None = None,
    stateful_risk: RiskManager | None = None,
) -> dict[str, Any]:
    effective_exec = _effective_exec_cfg(exec_cfg, policy)
    books = _build_books_from_row(row)
    if books is None:
        return {
            "policy": policy.name,
            "action": "no_trade",
            "reason": "missing_orderbook",
            "candidate_key": policy.candidate_key,
        }

    p_up = _candidate_probability(row, policy.candidate_key)
    if p_up is None:
        return {
            "policy": policy.name,
            "action": "no_trade",
            "reason": "missing_probability",
            "candidate_key": policy.candidate_key,
        }

    max_exposure = cap_exposure_by_balance(
        max_exposure_usd=float(risk_cfg.get("max_exposure_per_window_usd", balance)),
        balance_usd=float(balance),
        max_balance_fraction_per_trade=float(risk_cfg.get("max_balance_fraction_per_trade", 1.0)),
    )
    seconds_to_expiry = float(_to_float(row.get("seconds_to_expiry"), 300.0) or 300.0)
    decision: SignalDecision = decide_trade(
        market_slug=str(row.get("market_slug") or row.get("market_id") or "redecision_market"),
        books=books,
        p_up=float(p_up),
        min_edge_to_trade=float(effective_exec.get("min_edge_to_trade", 0.004)),
        min_edge_for_taker=float(effective_exec.get("min_edge_for_taker", 0.009)),
        max_exposure_usd=float(max_exposure),
        maker_preference=bool(effective_exec.get("maker_preference", True)),
        fee_cfg=fee_cfg,
        taker_slippage_bps=float(effective_exec.get("slippage_bps_taker", 12.0)),
        maker_slippage_bps=float(effective_exec.get("slippage_bps_maker", 3.0)),
        taker_fee_bps_by_token={
            books.up.token_id: float(_to_float(row.get("up_taker_fee_bps"), fee_cfg.taker_fee_bps) or fee_cfg.taker_fee_bps),
            books.down.token_id: float(_to_float(row.get("down_taker_fee_bps"), fee_cfg.taker_fee_bps) or fee_cfg.taker_fee_bps),
        },
        maker_fill_probability=float(effective_exec.get("maker_fill_probability", 0.65)),
        maker_fill_probability_by_token=_maker_fill_probability_map(books, seconds_to_expiry, effective_exec),
        maker_ev_advantage_required=float(effective_exec.get("maker_ev_advantage_required", 0.0005)),
        allowed_order_types=list(effective_exec.get("allowed_order_types", ["maker", "taker"])),
        lock_side_to_prediction=bool(effective_exec.get("lock_side_to_prediction", True)),
        min_reward_to_risk_ratio=float(effective_exec.get("min_reward_to_risk_ratio", 0.0)),
        min_expected_roi_cash=effective_exec.get("min_expected_roi_cash"),
        min_breakeven_margin=effective_exec.get("min_breakeven_margin"),
        sizing_mode=str(effective_exec.get("sizing_mode", "edge_scaled")),
    )
    decision, confirmation_gate_payload = apply_confirmation_gate(
        decision=decision,
        books=books,
        candidate_models=row.get("candidate_models") if isinstance(row.get("candidate_models"), dict) else {},
        gate_cfg=effective_exec.get("confirmation_gate", {}),
    )
    runtime_layers: dict[str, Any] = {}
    if decision.intent is not None:
        decision, runtime_layers = _apply_runtime_risk_layers(
            decision=decision,
            row=row,
            books=books,
            fee_cfg=fee_cfg,
            exec_cfg=effective_exec,
            risk_cfg=risk_cfg,
            base_exposure_usd=float(max_exposure),
            available_cash=float(balance),
            peak_cash=float(peak_balance if peak_balance is not None else balance),
            created_at=row.get("created_at") or row.get("ts_utc"),
            stateful_risk=stateful_risk,
        )

    out: dict[str, Any] = {
        "policy": policy.name,
        "candidate_key": policy.candidate_key,
        "market_id": row.get("market_id"),
        "market_slug": row.get("market_slug"),
        "created_at": row.get("created_at") or row.get("ts_utc"),
        "p_up": float(p_up),
        "action": decision.action,
        "reason": decision.reason,
        "best_edge": float(decision.best_edge),
        "score_mode": decision.score_mode,
        "score_value": decision.score_value,
        "expected_roi_cash": decision.expected_roi_cash,
        "breakeven_probability": decision.breakeven_probability,
        "breakeven_margin": decision.breakeven_margin,
        "fill_probability": decision.fill_probability,
        "ev_executable": decision.ev_executable,
        "ev_fill": decision.ev_fill,
        "decision_cash_required": decision.cash_required,
        **confirmation_gate_payload,
        **runtime_layers,
    }
    if decision.intent is None:
        return out

    pnl = _pnl_if_filled(decision.intent, row, books, fee_cfg, effective_exec)
    paper_sim = _paper_simulated_pnl(decision.intent, row, books, fee_cfg, effective_exec)
    out.update(
        {
            "order_type": decision.intent.order_type,
            "token_id": decision.intent.token_id,
            "selected_side": pnl.get("selected_side"),
            "price": float(decision.intent.price),
            "size": float(decision.intent.size),
            "expected_edge": float(decision.intent.expected_edge),
            "actual_side": pnl.get("actual_side"),
            "pnl_if_filled": pnl.get("pnl"),
            "win_if_filled": pnl.get("win"),
            "fee_if_filled": pnl.get("fee"),
            "slippage_if_filled": pnl.get("slippage"),
            "cash_required_if_filled": pnl.get("cash_required"),
            **paper_sim,
        }
    )
    return out


def _sort_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        for field_name in ("created_at", "ts_utc", "end_time", "market_id"):
            if row.get(field_name) is not None:
                return str(row.get(field_name))
        return ""

    return sorted(list(rows), key=key)


def run_redecision_replay(
    rows: Sequence[dict[str, Any]],
    *,
    policies: Sequence[RedecisionPolicy],
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    config: RedecisionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RedecisionConfig()
    ordered = _sort_rows(rows)
    policy_summaries: dict[str, dict[str, Any]] = {}

    for policy in policies:
        balance = float(cfg.initial_balance)
        peak_balance = float(balance)
        stateful_risk = RiskManager(
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
        decisions: list[dict[str, Any]] = []
        reason_counts: Counter[str] = Counter()
        order_type_counts: Counter[str] = Counter()
        trades = 0
        filled_trades = 0
        wins = 0
        losses = 0
        pnl_total = 0.0

        for row in ordered:
            created_at = _parse_ts(row.get("created_at") or row.get("ts_utc"))
            decision = redecide_row(
                row,
                policy=policy,
                fee_cfg=fee_cfg,
                exec_cfg=exec_cfg,
                risk_cfg=risk_cfg,
                balance=float(balance),
                peak_balance=float(peak_balance),
                stateful_risk=stateful_risk,
            )
            reason_counts[str(decision.get("reason") or "unknown")] += 1
            if decision.get("action") == "trade":
                cash_required = float(decision.get("cash_required_if_filled") or decision.get("decision_cash_required") or 0.0)
                if cash_required < float(cfg.min_trade_notional):
                    decision["action"] = "no_trade"
                    decision["reason"] = "trade_notional_too_small"
                    reason_counts["trade_notional_too_small"] += 1
                else:
                    ok = True
                    risk_reason = "ok"
                    if created_at is not None:
                        ok, risk_reason = stateful_risk.can_trade(created_at, exposure_usd=float(cash_required))
                    if not ok:
                        decision["action"] = "no_trade"
                        decision["reason"] = str(risk_reason)
                        reason_counts[str(risk_reason)] += 1
                    else:
                        trades += 1
                        order_type_counts[str(decision.get("order_type") or "UNKNOWN")] += 1
                        stateful_risk.on_trade_open()
                        fill_mode = str(getattr(cfg, "fill_mode", "assume_filled") or "assume_filled")
                        if fill_mode == "paper_sim":
                            filled = bool(decision.get("paper_sim_filled"))
                            pnl = decision.get("paper_sim_pnl") if filled else 0.0
                        else:
                            filled = bool(cfg.assume_filled)
                            pnl = decision.get("pnl_if_filled") if cfg.assume_filled else 0.0
                        filled_trades += int(filled)
                        if pnl is not None:
                            pnl_f = float(pnl)
                            pnl_total += pnl_f
                            balance += pnl_f
                            peak_balance = max(float(peak_balance), float(balance))
                            if filled and pnl_f > 0.0:
                                wins += 1
                            elif filled and pnl_f < 0.0:
                                losses += 1
                            stateful_risk.on_trade_close(pnl_f)
            if created_at is not None:
                decision["created_at"] = created_at.isoformat()
            if cfg.include_decisions:
                decisions.append(decision)

        policy_summaries[policy.name] = {
            "policy": asdict(policy),
            "rows": len(ordered),
            "fill_mode": str(getattr(cfg, "fill_mode", "assume_filled") or "assume_filled"),
            "initial_balance": float(cfg.initial_balance),
            "final_balance": float(balance),
            "net_pnl_if_filled": float(pnl_total),
            "net_pnl": float(pnl_total),
            "trades": int(trades),
            "submitted_trades": int(trades),
            "filled_trades": int(filled_trades),
            "fill_rate": float(filled_trades / trades) if trades > 0 else 0.0,
            "wins": int(wins),
            "losses": int(losses),
            "win_rate_if_filled": float(wins / filled_trades) if filled_trades > 0 else 0.0,
            "skip_or_reason_counts": dict(reason_counts),
            "order_type_counts": dict(order_type_counts),
            "decisions": decisions if cfg.include_decisions else None,
        }

    return {
        "config": asdict(cfg),
        "policy_count": len(policies),
        "row_count": len(ordered),
        "policies": policy_summaries,
    }
