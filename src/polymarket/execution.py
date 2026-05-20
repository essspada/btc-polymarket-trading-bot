from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.polymarket.clients.clob_client import ClobClient


@dataclass
class OrderIntent:
    market_slug: str
    token_id: str
    side: str
    order_type: str
    price: float
    size: float
    expected_edge: float


@dataclass
class ExecutionResult:
    accepted: bool
    mode: str
    fill_price: float
    fill_size: float
    note: str
    order_id: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class OrderReconcileResult:
    order_id: str
    status: str
    terminal: bool
    fill_price: float | None
    fill_size: float
    note: str
    raw_response: dict[str, Any] | None = None


@dataclass
class LiveOrderRecord:
    order_id: str
    market_id: str
    market_slug: str
    token_id: str
    side: str
    order_type: str
    price: float
    size: float
    expected_edge: float
    submitted_at: str
    status: str
    terminal: bool
    last_reconciled_at: str | None = None
    fill_price: float | None = None
    fill_size: float = 0.0
    raw_submit_response: dict[str, Any] | None = None
    raw_reconcile_response: dict[str, Any] | None = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _canonical_order_status(payload: dict[str, Any] | None) -> str:
    raw = str(
        (payload or {}).get("status")
        or (payload or {}).get("state")
        or (payload or {}).get("order_status")
        or (payload or {}).get("orderStatus")
        or ""
    ).strip().lower()
    if raw in {"open", "live", "active", "pending", "accepted", "booked", "unmatched"}:
        return "OPEN"
    if raw in {"filled", "matched", "executed", "complete", "completed"}:
        return "FILLED"
    if raw in {"partially_filled", "partial", "partially-filled", "partiallyfilled"}:
        return "PARTIALLY_FILLED"
    if raw in {"canceled", "cancelled"}:
        return "CANCELED"
    if raw in {"rejected", "failed", "error"}:
        return "REJECTED"
    if raw in {"expired"}:
        return "EXPIRED"
    if raw in {"submitted"}:
        return "SUBMITTED"
    return "UNKNOWN"


def _is_terminal_status(status: str) -> bool:
    return status in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}


def _extract_fill_size(payload: dict[str, Any] | None) -> float:
    for key in ("filled_size", "filledSize", "size_matched", "matched_size", "sizeMatched", "filled"):
        value = _safe_float((payload or {}).get(key))
        if value is not None:
            return float(value)
    return 0.0


def _extract_fill_price(payload: dict[str, Any] | None) -> float | None:
    for key in ("avg_fill_price", "avgFillPrice", "fill_price", "fillPrice", "average_price", "averagePrice", "price"):
        value = _safe_float((payload or {}).get(key))
        if value is not None:
            return float(value)
    return None


def load_live_order_records(path: Path) -> dict[str, LiveOrderRecord]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("orders", []) if isinstance(payload, dict) else []
    out: dict[str, LiveOrderRecord] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            record = LiveOrderRecord(**row)
        except Exception:
            continue
        out[record.order_id] = record
    return out


def save_live_order_records(path: Path, records: dict[str, LiveOrderRecord], now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now.isoformat(),
        "orders": [asdict(record) for record in records.values()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_live_order_record(
    *,
    market_id: str,
    intent: OrderIntent,
    result: ExecutionResult,
    now: datetime,
) -> LiveOrderRecord | None:
    if not result.accepted or result.order_id is None:
        return None
    status = _canonical_order_status(result.raw_response or {"status": result.note})
    return LiveOrderRecord(
        order_id=str(result.order_id),
        market_id=str(market_id),
        market_slug=str(intent.market_slug),
        token_id=str(intent.token_id),
        side=str(intent.side),
        order_type=str(intent.order_type),
        price=float(intent.price),
        size=float(intent.size),
        expected_edge=float(intent.expected_edge),
        submitted_at=now.isoformat(),
        status=status,
        terminal=_is_terminal_status(status),
        last_reconciled_at=now.isoformat(),
        fill_price=None,
        fill_size=0.0,
        raw_submit_response=dict(result.raw_response or {}),
        raw_reconcile_response=None,
    )


def apply_reconcile_result(record: LiveOrderRecord, reconcile: OrderReconcileResult, now: datetime) -> LiveOrderRecord:
    record.status = str(reconcile.status)
    record.terminal = bool(reconcile.terminal)
    record.last_reconciled_at = now.isoformat()
    record.fill_price = reconcile.fill_price
    record.fill_size = float(reconcile.fill_size)
    record.raw_reconcile_response = dict(reconcile.raw_response or {})
    return record


class ExecutionEngine:
    def __init__(
        self,
        live_trading: bool,
        require_confirm_live: bool = True,
        clob_client: ClobClient | None = None,
    ) -> None:
        self.live_trading = bool(live_trading)
        self.require_confirm_live = bool(require_confirm_live)
        self.clob_client = clob_client

    def _live_guard(self, confirm_live: bool) -> None:
        if not self.live_trading:
            raise RuntimeError("Live trading is disabled by config.")
        if os.getenv("LIVE_TRADING", "false").lower() != "true":
            raise RuntimeError("LIVE_TRADING env must be true for live mode.")
        if self.require_confirm_live and not confirm_live:
            raise RuntimeError("confirm_live flag is required in live mode.")

    def execute(
        self,
        intent: OrderIntent,
        now: datetime,
        confirm_live: bool = False,
    ) -> ExecutionResult:
        if not self.live_trading:
            return ExecutionResult(
                accepted=True,
                mode="SIM",
                fill_price=float(intent.price),
                fill_size=float(intent.size),
                note=f"simulated_{intent.order_type.lower()}_{intent.side.lower()}",
            )

        self._live_guard(confirm_live=confirm_live)
        if self.clob_client is None:
            raise RuntimeError("Authenticated ClobClient is required for live execution.")

        response = self.clob_client.place_order(
            {
                "market_slug": intent.market_slug,
                "token_id": intent.token_id,
                "side": intent.side,
                "order_type": intent.order_type,
                "price": float(intent.price),
                "size": float(intent.size),
            }
        )
        accepted = bool(response.get("accepted", False))
        order_id = response.get("order_id")
        return ExecutionResult(
            accepted=accepted,
            mode="LIVE_SUBMITTED" if accepted else "LIVE_REJECTED",
            fill_price=0.0,
            fill_size=0.0,
            note=str(response.get("status") or "submitted_unreconciled"),
            order_id=str(order_id) if order_id is not None else None,
            raw_response=dict(response),
        )

    def reconcile_order(
        self,
        order_id: str,
        now: datetime,
        confirm_live: bool = False,
    ) -> OrderReconcileResult:
        self._live_guard(confirm_live=confirm_live)
        if self.clob_client is None:
            raise RuntimeError("Authenticated ClobClient is required for live execution.")

        response = self.clob_client.get_order(str(order_id))
        status = _canonical_order_status(response)
        return OrderReconcileResult(
            order_id=str(order_id),
            status=status,
            terminal=_is_terminal_status(status),
            fill_price=_extract_fill_price(response),
            fill_size=_extract_fill_size(response),
            note=str(response.get("status") or response.get("state") or status),
            raw_response=dict(response),
        )
