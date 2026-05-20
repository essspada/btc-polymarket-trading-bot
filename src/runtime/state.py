"""Runtime dataclasses and on-disk state loaders.

The bot persists in-flight predictions, paper-trading positions, deferred
timing decisions, and bankroll snapshots between heartbeats so it can survive
restarts and resume cleanly. Everything that owns or restores that on-disk
state lives here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.strategy.timing_policy import DeferredTimingDecision

if TYPE_CHECKING:
    from src.polymarket.fees import FeeModelConfig


OUTPUT_SCHEMA_VERSION = 2


@dataclass
class PendingPrediction:
    market_id: str
    market_slug: str
    event_slug: str
    series_slug: str
    start_time: str
    end_time: str
    predicted_side: str
    p_up: float
    p_down: float
    confidence: float
    created_at: str
    seconds_from_window_start: float | None = None
    seconds_to_expiry: float | None = None
    schema_version: int = OUTPUT_SCHEMA_VERSION
    question: str | None = None
    resolution_source: str | None = None
    model_source: str = ""
    model_p_up: float | None = None
    market_p_up: float | None = None
    proxy_p_up: float | None = None
    blend_model_weight: float | None = None
    calibration_shift: float | None = None
    up_best_bid: float | None = None
    up_best_ask: float | None = None
    up_midpoint: float | None = None
    up_spread: float | None = None
    down_best_bid: float | None = None
    down_best_ask: float | None = None
    down_midpoint: float | None = None
    down_spread: float | None = None
    up_topk_imbalance: float | None = None
    down_topk_imbalance: float | None = None
    up_best_bid_size: float | None = None
    up_best_ask_size: float | None = None
    down_best_bid_size: float | None = None
    down_best_ask_size: float | None = None
    up_top3_bid_size: float | None = None
    up_top3_ask_size: float | None = None
    down_top3_bid_size: float | None = None
    down_top3_ask_size: float | None = None
    up_bid_level_count: int | None = None
    up_ask_level_count: int | None = None
    down_bid_level_count: int | None = None
    down_ask_level_count: int | None = None
    up_top5_bid_size: float | None = None
    up_top5_ask_size: float | None = None
    down_top5_bid_size: float | None = None
    down_top5_ask_size: float | None = None
    up_top10_bid_size: float | None = None
    up_top10_ask_size: float | None = None
    down_top10_bid_size: float | None = None
    down_top10_ask_size: float | None = None
    up_top5_imbalance: float | None = None
    up_top10_imbalance: float | None = None
    down_top5_imbalance: float | None = None
    down_top10_imbalance: float | None = None
    up_bid_vwap_top3: float | None = None
    up_ask_vwap_top3: float | None = None
    down_bid_vwap_top3: float | None = None
    down_ask_vwap_top3: float | None = None
    up_bid_depth_1c: float | None = None
    up_ask_depth_1c: float | None = None
    down_bid_depth_1c: float | None = None
    down_ask_depth_1c: float | None = None
    up_microprice: float | None = None
    down_microprice: float | None = None
    decision_action: str | None = None
    decision_reason: str | None = None
    decision_best_edge: float | None = None
    decision_order_type: str | None = None
    decision_expected_edge: float | None = None
    decision_score_mode: str | None = None
    decision_score_value: float | None = None
    decision_expected_roi_cash: float | None = None
    decision_breakeven_probability: float | None = None
    decision_breakeven_margin: float | None = None
    decision_fill_probability: float | None = None
    decision_ev_executable: float | None = None
    decision_ev_fill: float | None = None
    decision_cash_required: float | None = None
    decision_calibrated_p_side: float | None = None
    decision_calibrated_net_edge: float | None = None
    decision_calibrated_expected_roi_cash: float | None = None
    decision_calibrated_breakeven_margin: float | None = None
    decision_actionability_calibration_applied: bool | None = None
    decision_actionability_calibration_rejected: bool | None = None
    decision_actionability_calibration_reason: str | None = None
    decision_actionability_lower_bound_floor: float | None = None
    decision_actionability_lower_bound_sources: list[dict[str, Any]] | None = None
    decision_token_id: str | None = None
    book_quality_ok: bool | None = None
    book_quality_score: float | None = None
    book_quality_reason: str | None = None
    up_taker_fee_bps: float | None = None
    down_taker_fee_bps: float | None = None
    oracle_basis_bps: float | None = None
    oracle_spot_price: float | None = None
    oracle_price: float | None = None
    oracle_source: str | None = None
    oracle_external_feed: bool | None = None
    spot_price_now: float | None = None
    spot_window_open_price: float | None = None
    spot_return_bps_from_open: float | None = None
    spot_recent_return_1m_bps: float | None = None
    spot_recent_vol_5m_bps: float | None = None
    candidate_models: dict[str, Any] | None = None
    timing_policy_name: str | None = None
    timing_policy_source: str | None = None
    timing_policy_stage_delay: int | None = None
    timing_policy_stage_extra_edge: float | None = None
    timing_policy_reason: str | None = None
    timing_policy_entry_mode: str | None = None
    sizing_cap_enabled: bool = False
    sizing_cap_applied: bool = False
    sizing_cap_reason: str | None = None
    sizing_cap_ratio: float | None = None
    sizing_cap_below_min_trade: bool = False
    sizing_cap_raw_size: float | None = None
    sizing_cap_capped_size: float | None = None
    sizing_cap_raw_cash_required: float | None = None
    sizing_cap_capped_cash_required: float | None = None
    sizing_cap_min_trade_usd_after_cap: float | None = None
    sizing_cap_max_trade_usd: float | None = None
    sizing_cap_max_fraction_of_base_exposure: float | None = None
    sizing_cap_base_exposure_usd: float | None = None
    sizing_cap_effective_cap_usd: float | None = None
    x3_payload: dict[str, Any] | None = None
    adaptive_risk_payload: dict[str, Any] | None = None
    confirmation_gate_payload: dict[str, Any] | None = None
    resolved_ts_utc: str | None = None


@dataclass
class PendingPaperTrade:
    trade_id: str
    market_id: str
    market_slug: str
    event_slug: str
    series_slug: str
    start_time: str
    end_time: str
    created_at: str
    predicted_side: str
    p_up: float
    p_down: float
    model_source: str
    token_id: str
    order_type: str
    fill_price: float
    fill_size: float
    filled: bool
    fill_probability: float
    expected_edge: float
    up_token_id: str
    down_token_id: str
    up_best_ask: float
    down_best_ask: float
    schema_version: int = OUTPUT_SCHEMA_VERSION
    expected_roi_cash: float | None = None
    breakeven_probability: float | None = None
    breakeven_margin: float | None = None
    decision_score_mode: str | None = None
    decision_score_value: float | None = None
    decision_ev_executable: float | None = None
    decision_ev_fill: float | None = None
    decision_cash_required: float | None = None
    decision_calibrated_p_side: float | None = None
    decision_calibrated_net_edge: float | None = None
    decision_calibrated_expected_roi_cash: float | None = None
    decision_calibrated_breakeven_margin: float | None = None
    decision_actionability_calibration_applied: bool | None = None
    decision_actionability_calibration_rejected: bool | None = None
    decision_actionability_calibration_reason: str | None = None
    decision_actionability_lower_bound_floor: float | None = None
    decision_actionability_lower_bound_sources: list[dict[str, Any]] | None = None
    question: str | None = None
    resolution_source: str | None = None
    decision_reason: str | None = None
    up_topk_imbalance: float | None = None
    down_topk_imbalance: float | None = None
    up_best_bid_size: float | None = None
    up_best_ask_size: float | None = None
    down_best_bid_size: float | None = None
    down_best_ask_size: float | None = None
    up_top3_bid_size: float | None = None
    up_top3_ask_size: float | None = None
    down_top3_bid_size: float | None = None
    down_top3_ask_size: float | None = None
    up_bid_level_count: int | None = None
    up_ask_level_count: int | None = None
    down_bid_level_count: int | None = None
    down_ask_level_count: int | None = None
    up_top5_bid_size: float | None = None
    up_top5_ask_size: float | None = None
    down_top5_bid_size: float | None = None
    down_top5_ask_size: float | None = None
    up_top10_bid_size: float | None = None
    up_top10_ask_size: float | None = None
    down_top10_bid_size: float | None = None
    down_top10_ask_size: float | None = None
    up_top5_imbalance: float | None = None
    up_top10_imbalance: float | None = None
    down_top5_imbalance: float | None = None
    down_top10_imbalance: float | None = None
    up_bid_vwap_top3: float | None = None
    up_ask_vwap_top3: float | None = None
    down_bid_vwap_top3: float | None = None
    down_ask_vwap_top3: float | None = None
    up_bid_depth_1c: float | None = None
    up_ask_depth_1c: float | None = None
    down_bid_depth_1c: float | None = None
    down_ask_depth_1c: float | None = None
    up_microprice: float | None = None
    down_microprice: float | None = None
    up_taker_fee_bps: float | None = None
    down_taker_fee_bps: float | None = None
    selected_taker_fee_bps: float | None = None
    oracle_basis_bps: float | None = None
    spot_price_now: float | None = None
    spot_window_open_price: float | None = None
    spot_return_bps_from_open: float | None = None
    spot_recent_return_1m_bps: float | None = None
    spot_recent_vol_5m_bps: float | None = None
    candidate_models: dict[str, Any] | None = None
    timing_policy_name: str | None = None
    timing_policy_source: str | None = None
    timing_policy_stage_delay: int | None = None
    timing_policy_stage_extra_edge: float | None = None
    timing_policy_reason: str | None = None
    timing_policy_entry_mode: str | None = None
    sizing_cap_enabled: bool = False
    sizing_cap_applied: bool = False
    sizing_cap_reason: str | None = None
    sizing_cap_ratio: float | None = None
    sizing_cap_below_min_trade: bool = False
    sizing_cap_raw_size: float | None = None
    sizing_cap_capped_size: float | None = None
    sizing_cap_raw_cash_required: float | None = None
    sizing_cap_capped_cash_required: float | None = None
    sizing_cap_min_trade_usd_after_cap: float | None = None
    sizing_cap_max_trade_usd: float | None = None
    sizing_cap_max_fraction_of_base_exposure: float | None = None
    sizing_cap_base_exposure_usd: float | None = None
    sizing_cap_effective_cap_usd: float | None = None
    x3_payload: dict[str, Any] | None = None
    adaptive_risk_payload: dict[str, Any] | None = None
    confirmation_gate_payload: dict[str, Any] | None = None
    cash_required: float = 0.0


@dataclass
class PaperBankrollState:
    initial_balance: float
    available_cash: float
    reserved_cash: float = 0.0
    realized_pnl: float = 0.0
    realized_fees: float = 0.0
    realized_slippage: float = 0.0
    opened_trades: int = 0
    settled_trades: int = 0


def load_pending_predictions(path: Path) -> dict[str, PendingPrediction]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    pending: dict[str, PendingPrediction] = {}
    for row in payload.get("pending", []):
        if not isinstance(row, dict):
            continue
        row_local = dict(row)
        if "schema_version" not in row_local:
            row_local["schema_version"] = 1
        try:
            item = PendingPrediction(**row_local)
        except Exception:
            continue
        pending[item.market_id] = item
    return pending


def load_pending_paper_trades(path: Path) -> dict[str, PendingPaperTrade]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    pending: dict[str, PendingPaperTrade] = {}
    rows = payload.get("pending_trades", []) if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_local = dict(row)
        if "schema_version" not in row_local:
            row_local["schema_version"] = 1
        try:
            item = PendingPaperTrade(**row_local)
        except Exception:
            continue
        pending[item.market_id] = item
    return pending


def load_deferred_timing(path: Path) -> dict[str, DeferredTimingDecision]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    deferred: dict[str, DeferredTimingDecision] = {}
    rows = payload.get("deferred_timing", []) if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            item = DeferredTimingDecision(**row)
        except Exception:
            continue
        deferred[item.market_id] = item
    return deferred


def load_running_stats(outcomes_path: Path) -> tuple[set[str], int, int]:
    from src.runtime.persistence import read_jsonl

    resolved_ids: set[str] = set()
    total = 0
    correct = 0

    for row in read_jsonl(outcomes_path):
        market_id = str(row.get("market_id", ""))
        if market_id:
            resolved_ids.add(market_id)
        if "is_correct" in row:
            total += 1
            is_correct_raw = row.get("is_correct")
            is_correct = (
                is_correct_raw
                if isinstance(is_correct_raw, bool)
                else str(is_correct_raw).strip().lower() in {"1", "true", "yes"}
            )
            if is_correct:
                correct += 1

    return resolved_ids, total, correct


def pending_trade_is_active(trade: PendingPaperTrade) -> bool:
    return bool(getattr(trade, "filled", False)) and float(getattr(trade, "fill_size", 0.0) or 0.0) > 0.0


def pending_trade_reserved_cash(
    trade: PendingPaperTrade,
    *,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
) -> float:
    from src.runtime.sizing import cash_required_for_order

    if not pending_trade_is_active(trade):
        return 0.0
    if float(getattr(trade, "cash_required", 0.0) or 0.0) > 0.0:
        return float(trade.cash_required)
    return cash_required_for_order(
        price=float(trade.fill_price),
        size=float(trade.fill_size),
        order_type=str(trade.order_type),
        fee_cfg=fee_cfg,
        taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
        maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3.0)),
        taker_fee_bps=trade.selected_taker_fee_bps,
    )


def count_active_pending_paper_trades(pending_trades: dict[str, PendingPaperTrade]) -> int:
    return sum(1 for trade in pending_trades.values() if pending_trade_is_active(trade))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_paper_realized_totals(outcomes_path: Path) -> tuple[float, float, float, int]:
    from src.runtime.persistence import read_jsonl

    realized_pnl = 0.0
    realized_fees = 0.0
    realized_slippage = 0.0
    settled_trades = 0

    for row in read_jsonl(outcomes_path):
        trade_id = row.get("trade_id")
        if trade_id in {None, ""}:
            continue
        settled_trades += 1
        realized_pnl += _to_float(row.get("trade_net_pnl"), 0.0)
        realized_fees += _to_float(row.get("trade_fee"), 0.0)
        realized_slippage += _to_float(row.get("trade_slippage"), 0.0)

    return float(realized_pnl), float(realized_fees), float(realized_slippage), int(settled_trades)


def load_paper_bankroll_state(
    *,
    state_path: Path,
    outcomes_path: Path,
    pending_trades: dict[str, PendingPaperTrade],
    initial_balance: float,
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
) -> PaperBankrollState:
    payload: dict[str, Any] = {}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}

    state_raw = payload.get("paper_bankroll") if isinstance(payload, dict) else None
    if isinstance(state_raw, dict):
        row = dict(state_raw)
        row.setdefault("initial_balance", float(initial_balance))
        try:
            return PaperBankrollState(**row)
        except Exception:
            pass

    realized_pnl, realized_fees, realized_slippage, settled_trades = load_paper_realized_totals(outcomes_path)
    reserved_cash = 0.0
    for trade in pending_trades.values():
        reserved_cash += pending_trade_reserved_cash(
            trade,
            fee_cfg=fee_cfg,
            exec_cfg=exec_cfg,
        )
    available_cash = max(0.0, float(initial_balance) + float(realized_pnl) - float(reserved_cash))
    return PaperBankrollState(
        initial_balance=float(initial_balance),
        available_cash=float(available_cash),
        reserved_cash=float(reserved_cash),
        realized_pnl=float(realized_pnl),
        realized_fees=float(realized_fees),
        realized_slippage=float(realized_slippage),
        opened_trades=len(pending_trades) + int(settled_trades),
        settled_trades=int(settled_trades),
    )
