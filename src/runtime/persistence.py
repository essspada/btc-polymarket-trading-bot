"""JSON/JSONL persistence and monitor-observation logging."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.polymarket.market_discovery import DiscoveredMarket
from src.polymarket.orderbook import MarketBooks
from src.runtime.state import OUTPUT_SCHEMA_VERSION
from src.strategy.timing_policy import TimingPolicyStage, timing_policy_entry_mode


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
                if max_rows is not None and max_rows > 0 and len(rows) > max_rows:
                    rows = rows[-max_rows:]
    return rows


def append_timing_policy_event(
    path: Path | None,
    *,
    mode: str,
    now: datetime,
    market: DiscoveredMarket,
    seconds_from_start: float,
    selection: dict[str, Any] | None,
    action: str,
    current_stage_index: int | None,
    late_start_grace_seconds: float = 0.0,
    probs: dict[str, Any] | None = None,
) -> None:
    if path is None:
        return
    stage = selection.get("stage") if isinstance(selection, dict) else None
    payload: dict[str, Any] = {
        "ts_utc": now.isoformat(),
        "mode": str(mode),
        "event": str(action),
        "market_id": market.market_id,
        "market_slug": market.market_slug,
        "event_slug": market.event_slug,
        "series_slug": str(market.series_slug or ""),
        "seconds_from_window_start": float(seconds_from_start),
        "current_stage_index": (int(current_stage_index) if current_stage_index is not None else None),
        "reason": (str(selection.get("reason")) if isinstance(selection, dict) and selection.get("reason") is not None else None),
        "timing_policy_name": (
            str(probs.get("timing_policy_name"))
            if isinstance(probs, dict) and probs.get("timing_policy_name") is not None
            else None
        ),
        "timing_policy_source": (
            str(probs.get("timing_policy_source"))
            if isinstance(probs, dict) and probs.get("timing_policy_source") is not None
            else None
        ),
        "timing_policy_stage_delay": (
            int(probs.get("timing_policy_stage_delay"))
            if isinstance(probs, dict) and probs.get("timing_policy_stage_delay") is not None
            else None
        ),
        "timing_policy_stage_extra_edge": (
            float(probs.get("timing_policy_stage_extra_edge"))
            if isinstance(probs, dict) and probs.get("timing_policy_stage_extra_edge") is not None
            else None
        ),
        "timing_policy_entry_mode": (
            str(probs.get("timing_policy_entry_mode"))
            if isinstance(probs, dict) and probs.get("timing_policy_entry_mode") is not None
            else None
        ),
        "next_stage_index": (
            int(selection.get("next_stage_index"))
            if isinstance(selection, dict) and selection.get("next_stage_index") is not None
            else None
        ),
    }
    if isinstance(stage, TimingPolicyStage):
        payload["stage_candidate_key"] = str(stage.candidate_key)
        payload["stage_delay_seconds"] = int(stage.delay_seconds)
        payload["stage_extra_edge"] = float(stage.extra_edge)
        if payload.get("timing_policy_entry_mode") is None:
            payload["timing_policy_entry_mode"] = timing_policy_entry_mode(
                stage=stage,
                current_stage_index=current_stage_index,
                seconds_from_start=seconds_from_start,
                late_start_grace_seconds=late_start_grace_seconds,
            )
    if isinstance(probs, dict):
        payload["p_up"] = probs.get("p_up")
        payload["predicted_side"] = probs.get("predicted_side")
    append_jsonl(path, payload)


def append_monitor_observation(
    path: Path | None,
    *,
    now: datetime,
    market: DiscoveredMarket,
    seconds_from_start: float,
    seconds_to_expiry: float,
    model_source: str,
    timing_policy_active: bool,
    timing_action: str | None,
    timing_selection: dict[str, Any] | None,
    current_stage_index: int | None,
    books: MarketBooks | None,
    raw_orderbook: dict[str, Any] | None,
    quality: Any | None,
    fee_bps_map: dict[str, float] | None,
    spot_ctx: Any | None,
    oracle_basis: Any | None,
    oracle_source: str,
    oracle_external_feed: bool,
    model_weight: float | None,
    calibration_shift: float | None,
    probs: dict[str, Any] | None,
    candidate_models_payload: dict[str, Any] | None,
    book_error: str | None = None,
    model_error: str | None = None,
) -> None:
    from src.runtime.snapshots import book_depth_payload, spot_context_payload

    if path is None:
        return
    stage = timing_selection.get("stage") if isinstance(timing_selection, dict) else None
    payload: dict[str, Any] = {
        "schema": "monitor_observation_v1",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "ts_utc": now.isoformat(),
        "mode": "monitor-5m",
        "market_id": market.market_id,
        "market_slug": market.market_slug,
        "event_slug": market.event_slug,
        "series_slug": str(market.series_slug or ""),
        "question": market.question,
        "resolution_source": market.resolution_source,
        "start_time": market.start_time.isoformat(),
        "end_time": market.end_time.isoformat(),
        "accepting_orders": bool(market.accepting_orders),
        "seconds_from_window_start": float(seconds_from_start),
        "seconds_to_expiry": float(seconds_to_expiry),
        "model_source_config": str(model_source),
        "timing_policy_active": bool(timing_policy_active),
        "timing_action": str(timing_action) if timing_action is not None else None,
        "timing_reason": (
            str(timing_selection.get("reason"))
            if isinstance(timing_selection, dict) and timing_selection.get("reason") is not None
            else None
        ),
        "current_stage_index": int(current_stage_index) if current_stage_index is not None else None,
        "next_stage_index": (
            int(timing_selection.get("next_stage_index"))
            if isinstance(timing_selection, dict) and timing_selection.get("next_stage_index") is not None
            else None
        ),
        "stage_candidate_key": str(stage.candidate_key) if isinstance(stage, TimingPolicyStage) else None,
        "stage_delay_seconds": int(stage.delay_seconds) if isinstance(stage, TimingPolicyStage) else None,
        "stage_extra_edge": float(stage.extra_edge) if isinstance(stage, TimingPolicyStage) else None,
        "book_available": books is not None,
        "book_error": book_error,
        "model_error": model_error,
        "book_quality_ok": bool(getattr(quality, "ok", False)) if quality is not None else None,
        "book_quality_score": float(getattr(quality, "score", 0.0)) if quality is not None else None,
        "book_quality_reason": getattr(quality, "reason", None) if quality is not None else None,
        "book_quality_flags": list(getattr(quality, "flags", []) or []) if quality is not None else None,
        "book_quality_metrics": dict(getattr(quality, "metrics", {}) or {}) if quality is not None else None,
        "spot_context": spot_context_payload(spot_ctx),
        "oracle_basis_bps": getattr(oracle_basis, "basis_bps", None) if oracle_basis is not None else None,
        "oracle_spot_price": getattr(oracle_basis, "spot_price", None) if oracle_basis is not None else None,
        "oracle_price": getattr(oracle_basis, "oracle_price", None) if oracle_basis is not None else None,
        "oracle_source": oracle_source,
        "oracle_external_feed": bool(oracle_external_feed),
        "model_weight": model_weight,
        "calibration_shift": calibration_shift,
        "p_up": probs.get("p_up") if isinstance(probs, dict) else None,
        "p_down": probs.get("p_down") if isinstance(probs, dict) else None,
        "predicted_side": probs.get("predicted_side") if isinstance(probs, dict) else None,
        "confidence": probs.get("confidence") if isinstance(probs, dict) else None,
        "p_pre_calibration": probs.get("p_pre_calibration") if isinstance(probs, dict) else None,
        "model_used": probs.get("model_used") if isinstance(probs, dict) else None,
        "model_p_up": probs.get("model_p_up") if isinstance(probs, dict) else None,
        "market_p_up": probs.get("market_p_up") if isinstance(probs, dict) else None,
        "proxy_p_up": probs.get("proxy_p_up") if isinstance(probs, dict) else None,
        "candidate_probabilities": probs.get("candidate_probabilities") if isinstance(probs, dict) else None,
        "candidate_models": candidate_models_payload,
        "taker_fee_bps_by_token": dict(fee_bps_map or {}),
        "raw_orderbook": raw_orderbook,
    }
    if books is not None:
        payload.update(
            {
                "up_best_bid": books.up.best_bid,
                "up_best_ask": books.up.best_ask,
                "up_midpoint": books.up.midpoint,
                "up_spread": books.up.spread,
                "up_best_bid_size": books.up.best_bid_size,
                "up_best_ask_size": books.up.best_ask_size,
                "up_top3_bid_size": books.up.top3_bid_size,
                "up_top3_ask_size": books.up.top3_ask_size,
                "down_best_bid": books.down.best_bid,
                "down_best_ask": books.down.best_ask,
                "down_midpoint": books.down.midpoint,
                "down_spread": books.down.spread,
                "down_best_bid_size": books.down.best_bid_size,
                "down_best_ask_size": books.down.best_ask_size,
                "down_top3_bid_size": books.down.top3_bid_size,
                "down_top3_ask_size": books.down.top3_ask_size,
                **book_depth_payload(books),
            }
        )
        if fee_bps_map:
            payload["up_taker_fee_bps"] = fee_bps_map.get(books.up.token_id)
            payload["down_taker_fee_bps"] = fee_bps_map.get(books.down.token_id)
    append_jsonl(path, payload)
