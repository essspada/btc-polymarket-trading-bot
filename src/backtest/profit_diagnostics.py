from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from src.backtest.redecision import read_jsonl


@dataclass(frozen=True)
class NumericBin:
    label: str
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value < self.high


CONFIDENCE_BINS = (
    NumericBin("0.50-0.60", 0.50, 0.60),
    NumericBin("0.60-0.70", 0.60, 0.70),
    NumericBin("0.70-0.80", 0.70, 0.80),
    NumericBin("0.80-0.90", 0.80, 0.90),
    NumericBin("0.90-0.95", 0.90, 0.95),
    NumericBin("0.95-1.00", 0.95, 1.000001),
)

EDGE_BINS = (
    NumericBin("<0", -999.0, 0.0),
    NumericBin("0.00-0.01", 0.0, 0.01),
    NumericBin("0.01-0.03", 0.01, 0.03),
    NumericBin("0.03-0.07", 0.03, 0.07),
    NumericBin("0.07-0.15", 0.07, 0.15),
    NumericBin(">=0.15", 0.15, 999.0),
)

PRICE_BINS = (
    NumericBin("0.00-0.10", 0.0, 0.10),
    NumericBin("0.10-0.25", 0.10, 0.25),
    NumericBin("0.25-0.50", 0.25, 0.50),
    NumericBin("0.50-0.75", 0.50, 0.75),
    NumericBin("0.75-0.90", 0.75, 0.90),
    NumericBin("0.90-1.00", 0.90, 1.000001),
)


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _bin(value: Any, bins: Sequence[NumericBin], missing: str = "missing") -> str:
    val = _to_float(value)
    if val is None:
        return missing
    for item in bins:
        if item.contains(float(val)):
            return item.label
    return "out_of_range"


def _side(row: dict[str, Any]) -> str:
    return str(row.get("predicted_side") or "").strip().lower() or "missing"


def _actual(row: dict[str, Any]) -> str:
    return str(row.get("actual_side") or "").strip().lower() or "missing"


def _is_signal_correct(row: dict[str, Any]) -> bool | None:
    side = _side(row)
    actual = _actual(row)
    if side not in {"up", "down"} or actual not in {"up", "down"}:
        return None
    return side == actual


def _selected_entry_price(row: dict[str, Any]) -> float | None:
    side = _side(row)
    order_type = str(row.get("decision_order_type") or row.get("trade_order_type") or "").strip().upper()
    if side == "up":
        if order_type == "TAKER":
            return _to_float(row.get("up_best_ask"))
        return _to_float(row.get("up_best_bid"))
    if side == "down":
        if order_type == "TAKER":
            return _to_float(row.get("down_best_ask"))
        return _to_float(row.get("down_best_bid"))
    return None


def _init_group() -> dict[str, Any]:
    return {
        "rows": 0,
        "signals": 0,
        "signal_correct": 0,
        "proposed": 0,
        "filled": 0,
        "filled_wins": 0,
        "filled_losses": 0,
        "pnl": 0.0,
        "confidence_sum": 0.0,
        "confidence_n": 0,
        "edge_sum": 0.0,
        "edge_n": 0,
    }


def _add_row(group: dict[str, Any], row: dict[str, Any]) -> None:
    group["rows"] += 1
    correct = _is_signal_correct(row)
    if correct is not None:
        group["signals"] += 1
        group["signal_correct"] += int(correct)

    proposed = str(row.get("decision_action") or "").strip().lower() == "trade"
    group["proposed"] += int(proposed)
    filled = bool(row.get("trade_filled"))
    group["filled"] += int(filled)
    if filled:
        pnl = float(_to_float(row.get("trade_net_pnl"), 0.0) or 0.0)
        group["pnl"] += pnl
        group["filled_wins"] += int(pnl > 0.0)
        group["filled_losses"] += int(pnl < 0.0)

    conf = _to_float(row.get("confidence"))
    if conf is not None:
        group["confidence_sum"] += float(conf)
        group["confidence_n"] += 1
    edge = _to_float(row.get("decision_best_edge"))
    if edge is not None:
        group["edge_sum"] += float(edge)
        group["edge_n"] += 1


def _finalize_group(group: dict[str, Any]) -> dict[str, Any]:
    out = dict(group)
    out["signal_accuracy"] = _safe_div(float(group["signal_correct"]), float(group["signals"]))
    out["fill_rate"] = _safe_div(float(group["filled"]), float(group["proposed"]))
    out["filled_win_rate"] = _safe_div(float(group["filled_wins"]), float(group["filled"]))
    out["avg_pnl_per_filled"] = _safe_div(float(group["pnl"]), float(group["filled"]))
    out["avg_confidence"] = _safe_div(float(group["confidence_sum"]), float(group["confidence_n"]))
    out["avg_edge"] = _safe_div(float(group["edge_sum"]), float(group["edge_n"]))
    return out


def _group_by(rows: Iterable[dict[str, Any]], key_name: str, key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(_init_group)
    for row in rows:
        key = str(key_fn(row))
        groups[key][key_name] = key
        _add_row(groups[key], row)
    return {key: _finalize_group(value) for key, value in sorted(groups.items())}


def compute_profit_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows_list = list(rows)
    overall = _init_group()
    proposed_filled = _init_group()
    proposed_unfilled = _init_group()
    no_trade = _init_group()
    decision_reasons: Counter[str] = Counter()

    for row in rows_list:
        _add_row(overall, row)
        reason = str(row.get("decision_reason") or "missing")
        decision_reasons[reason] += 1
        proposed = str(row.get("decision_action") or "").strip().lower() == "trade"
        filled = bool(row.get("trade_filled"))
        if proposed and filled:
            _add_row(proposed_filled, row)
        elif proposed:
            _add_row(proposed_unfilled, row)
        else:
            _add_row(no_trade, row)

    filled_final = _finalize_group(proposed_filled)
    unfilled_final = _finalize_group(proposed_unfilled)
    adverse_gap = float(filled_final["signal_accuracy"] - unfilled_final["signal_accuracy"])

    group_specs = {
        "by_side": lambda r: _side(r),
        "by_confidence_bin": lambda r: _bin(r.get("confidence"), CONFIDENCE_BINS),
        "by_edge_bin": lambda r: _bin(r.get("decision_best_edge"), EDGE_BINS),
        "by_entry_price_bin": lambda r: _bin(_selected_entry_price(r), PRICE_BINS),
        "by_timing_stage": lambda r: r.get("timing_policy_stage_delay") if r.get("timing_policy_stage_delay") is not None else "missing",
        "by_timing_reason": lambda r: r.get("timing_policy_reason") or "missing",
        "by_entry_mode": lambda r: r.get("timing_policy_entry_mode") or "missing",
    }
    groups = {name: _group_by(rows_list, name, fn) for name, fn in group_specs.items()}

    flat_groups: list[dict[str, Any]] = []
    for group_name, group_values in groups.items():
        for key, stats in group_values.items():
            item = dict(stats)
            item["group"] = group_name
            item["key"] = key
            flat_groups.append(item)

    worst_groups = sorted(
        [item for item in flat_groups if int(item.get("filled", 0)) > 0],
        key=lambda item: float(item.get("pnl", 0.0)),
    )[:15]
    best_groups = sorted(
        [item for item in flat_groups if int(item.get("filled", 0)) > 0],
        key=lambda item: float(item.get("pnl", 0.0)),
        reverse=True,
    )[:15]

    return {
        "rows": len(rows_list),
        "overall": _finalize_group(overall),
        "proposed_filled": filled_final,
        "proposed_unfilled": unfilled_final,
        "no_trade": _finalize_group(no_trade),
        "adverse_selection_signal_accuracy_gap_filled_minus_unfilled": adverse_gap,
        "decision_reason_counts": dict(decision_reasons),
        "groups": groups,
        "worst_groups": worst_groups,
        "best_groups": best_groups,
    }


def compute_profit_diagnostics_from_jsonl(path: str) -> dict[str, Any]:
    return compute_profit_diagnostics(read_jsonl(path))
