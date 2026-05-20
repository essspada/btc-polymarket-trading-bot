from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.strategy.decision_brain import ExpertSignal, build_shadow_expert_signals


@dataclass(frozen=True)
class NumericBin:
    label: str
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value < self.high


CONFIDENCE_BINS = (
    NumericBin("missing", -1.0, -1.0),
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

VOL_BINS = (
    NumericBin("missing", -1.0, -1.0),
    NumericBin("0-2bps", 0.0, 2.0),
    NumericBin("2-5bps", 2.0, 5.0),
    NumericBin("5-8bps", 5.0, 8.0),
    NumericBin("8-12bps", 8.0, 12.0),
    NumericBin(">=12bps", 12.0, 999.0),
)

RETURN_BINS = (
    NumericBin("<-10bps", -999.0, -10.0),
    NumericBin("-10--5bps", -10.0, -5.0),
    NumericBin("-5--2bps", -5.0, -2.0),
    NumericBin("-2-2bps", -2.0, 2.0),
    NumericBin("2-5bps", 2.0, 5.0),
    NumericBin("5-10bps", 5.0, 10.0),
    NumericBin(">=10bps", 10.0, 999.0),
)

DEFAULT_SCOPES: Mapping[str, tuple[str, ...]] = {
    "expert": (),
    "expert_side": ("side",),
    "expert_timing": ("timing_reason",),
    "expert_side_timing": ("side", "timing_reason"),
    "expert_regime": ("side", "confidence_bin", "edge_bin", "entry_price_bin", "timing_reason", "vol_bin"),
}


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


def _side(row: Mapping[str, Any]) -> str:
    side = str(row.get("predicted_side") or "").strip().lower()
    if side in {"up", "down"}:
        return side
    p_up = _to_float(row.get("p_up"))
    if p_up is None:
        return "missing"
    return "up" if float(p_up) >= 0.5 else "down"


def _selected_entry_price(row: Mapping[str, Any]) -> float | None:
    side = _side(row)
    order_type = str(row.get("decision_order_type") or row.get("trade_order_type") or "").strip().upper()
    if side == "up":
        return _to_float(row.get("up_best_ask" if order_type == "TAKER" else "up_best_bid"))
    if side == "down":
        return _to_float(row.get("down_best_ask" if order_type == "TAKER" else "down_best_bid"))
    return None


def _signal_correct(row: Mapping[str, Any]) -> bool | None:
    side = _side(row)
    actual = str(row.get("actual_side") or "").strip().lower()
    if side not in {"up", "down"} or actual not in {"up", "down"}:
        return None
    return side == actual


def _cash_required(row: Mapping[str, Any]) -> float:
    for key in (
        "trade_cash_required",
        "decision_cash_required",
        "sizing_cap_capped_cash_required",
        "sizing_cap_raw_cash_required",
    ):
        value = _to_float(row.get(key))
        if value is not None and value > 0.0:
            return float(value)

    size = _to_float(row.get("trade_fill_size"))
    price = _selected_entry_price(row)
    fee = _to_float(row.get("trade_fee"), 0.0) or 0.0
    slippage = _to_float(row.get("trade_slippage"), 0.0) or 0.0
    if size is not None and price is not None and size > 0.0 and price > 0.0:
        return float(size * price + fee + slippage)
    return 0.0


def extract_regime_tags(row: Mapping[str, Any]) -> dict[str, str]:
    """Return stable, low-cardinality tags used by reliability memory.

    The tags intentionally avoid raw timestamps and raw prices. This keeps the
    table useful for replay instead of memorizing one-off market IDs.
    """

    return {
        "side": _side(row),
        "confidence_bin": _bin(row.get("confidence"), CONFIDENCE_BINS),
        "edge_bin": _bin(row.get("decision_best_edge"), EDGE_BINS),
        "entry_price_bin": _bin(_selected_entry_price(row), PRICE_BINS),
        "timing_stage": str(row.get("timing_policy_stage_delay") if row.get("timing_policy_stage_delay") is not None else "missing"),
        "timing_reason": str(row.get("timing_policy_reason") or "missing"),
        "entry_mode": str(row.get("timing_policy_entry_mode") or "missing"),
        "order_type": str(row.get("decision_order_type") or row.get("trade_order_type") or "missing").strip().lower() or "missing",
        "vol_bin": _bin(row.get("spot_recent_vol_5m_bps"), VOL_BINS),
        "return_1m_bin": _bin(row.get("spot_recent_return_1m_bps"), RETURN_BINS),
        "book_quality": "ok" if row.get("book_quality_ok") is True else ("bad" if row.get("book_quality_ok") is False else "missing"),
    }


@dataclass
class ReliabilityStats:
    key: str
    expert: str
    event: str
    scope: str
    tags: dict[str, str] = field(default_factory=dict)
    rows: int = 0
    proposed: int = 0
    filled: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    cash: float = 0.0
    signal_rows: int = 0
    signal_correct: int = 0
    veto_reasons: Counter[str] = field(default_factory=Counter)

    def observe(self, row: Mapping[str, Any], signal: ExpertSignal) -> None:
        self.rows += 1
        proposed = str(row.get("decision_action") or "").strip().lower() == "trade"
        self.proposed += int(proposed)
        correct = _signal_correct(row)
        if correct is not None:
            self.signal_rows += 1
            self.signal_correct += int(correct)
        if signal.veto:
            self.veto_reasons[str(signal.veto_reason or "veto")] += 1

        if not bool(row.get("trade_filled")):
            return
        pnl = float(_to_float(row.get("trade_net_pnl"), 0.0) or 0.0)
        self.filled += 1
        self.wins += int(pnl > 0.0)
        self.losses += int(pnl < 0.0)
        self.pnl += pnl
        self.cash += _cash_required(row)

    def finalize(
        self,
        *,
        prior_filled: float,
        prior_roi: float,
        prior_win_rate: float,
        min_filled_for_label: int,
        toxic_roi_threshold: float,
        profitable_roi_threshold: float,
    ) -> dict[str, Any]:
        raw_roi = _safe_div(float(self.pnl), float(self.cash))
        support = _safe_div(float(self.filled), float(self.filled) + float(prior_filled))
        shrunk_roi = raw_roi * support + float(prior_roi) * (1.0 - support)
        shrunk_win_rate = _safe_div(float(self.wins) + float(prior_win_rate) * float(prior_filled), float(self.filled) + float(prior_filled))

        if self.filled < min_filled_for_label:
            label = "insufficient_support"
        elif shrunk_roi <= toxic_roi_threshold:
            label = "toxic"
        elif shrunk_roi >= profitable_roi_threshold:
            label = "promising"
        else:
            label = "neutral"

        return {
            "key": self.key,
            "expert": self.expert,
            "event": self.event,
            "scope": self.scope,
            "tags": dict(self.tags),
            "rows": int(self.rows),
            "proposed": int(self.proposed),
            "filled": int(self.filled),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "pnl": float(self.pnl),
            "cash": float(self.cash),
            "raw_roi": float(raw_roi),
            "support": float(support),
            "shrunk_roi": float(shrunk_roi),
            "raw_win_rate": _safe_div(float(self.wins), float(self.filled)),
            "shrunk_win_rate": float(shrunk_win_rate),
            "avg_pnl_per_filled": _safe_div(float(self.pnl), float(self.filled)),
            "fill_rate": _safe_div(float(self.filled), float(self.proposed)),
            "signal_accuracy": _safe_div(float(self.signal_correct), float(self.signal_rows)),
            "risk_label": label,
            "veto_reasons": dict(self.veto_reasons),
        }


def _make_key(expert: str, event: str, scope: str, tags: Mapping[str, str], fields: Sequence[str]) -> tuple[str, dict[str, str]]:
    selected = {field: str(tags.get(field, "missing")) for field in fields}
    parts = [f"expert={expert}", f"event={event}", f"scope={scope}"]
    parts.extend(f"{field}={selected[field]}" for field in fields)
    return "|".join(parts), selected


def _new_stats(expert: str, event: str, scope: str, key: str, tags: Mapping[str, str]) -> ReliabilityStats:
    return ReliabilityStats(key=key, expert=expert, event=event, scope=scope, tags=dict(tags))


def build_reliability_memory(
    rows: Iterable[Mapping[str, Any]],
    *,
    scopes: Mapping[str, tuple[str, ...]] | None = None,
    include_pass: bool = False,
    prior_filled: float = 20.0,
    prior_roi: float = 0.0,
    prior_win_rate: float = 0.50,
    min_filled_for_label: int = 3,
    toxic_roi_threshold: float = -0.05,
    profitable_roi_threshold: float = 0.05,
) -> dict[str, Any]:
    rows_list = list(rows)
    active_scopes = scopes or DEFAULT_SCOPES
    stats: dict[str, ReliabilityStats] = {}
    global_stats = ReliabilityStats(key="global", expert="global", event="all", scope="global")

    for row in rows_list:
        tags = extract_regime_tags(row)
        synthetic_global_signal = ExpertSignal(name="global", veto=False)
        global_stats.observe(row, synthetic_global_signal)
        for signal in build_shadow_expert_signals(dict(row)):
            event = "veto" if signal.veto else "pass"
            if event == "pass" and not include_pass:
                continue
            for scope, fields in active_scopes.items():
                key, selected_tags = _make_key(signal.name, event, scope, tags, fields)
                item = stats.get(key)
                if item is None:
                    item = _new_stats(signal.name, event, scope, key, selected_tags)
                    stats[key] = item
                item.observe(row, signal)

    finalize_kwargs = {
        "prior_filled": float(prior_filled),
        "prior_roi": float(prior_roi),
        "prior_win_rate": float(prior_win_rate),
        "min_filled_for_label": int(min_filled_for_label),
        "toxic_roi_threshold": float(toxic_roi_threshold),
        "profitable_roi_threshold": float(profitable_roi_threshold),
    }
    entries = [item.finalize(**finalize_kwargs) for item in stats.values()]
    entries.sort(key=lambda item: (str(item["expert"]), str(item["event"]), str(item["scope"]), str(item["key"])))

    filled_entries = [item for item in entries if int(item.get("filled", 0)) > 0]
    top_toxic = sorted(filled_entries, key=lambda item: (float(item["shrunk_roi"]), -int(item["filled"])))[:30]
    top_promising = sorted(filled_entries, key=lambda item: (float(item["shrunk_roi"]), int(item["filled"])), reverse=True)[:30]

    return {
        "rows": len(rows_list),
        "config": {
            "include_pass": bool(include_pass),
            "prior_filled": float(prior_filled),
            "prior_roi": float(prior_roi),
            "prior_win_rate": float(prior_win_rate),
            "min_filled_for_label": int(min_filled_for_label),
            "toxic_roi_threshold": float(toxic_roi_threshold),
            "profitable_roi_threshold": float(profitable_roi_threshold),
            "scopes": {name: list(fields) for name, fields in active_scopes.items()},
        },
        "global": global_stats.finalize(**finalize_kwargs),
        "entries": entries,
        "top_toxic": top_toxic,
        "top_promising": top_promising,
    }


def score_row_reliability(
    row: Mapping[str, Any],
    memory: Mapping[str, Any],
    *,
    scope_preference: Sequence[str] = ("expert_regime", "expert_side_timing", "expert_timing", "expert_side", "expert"),
    actionable_labels: Sequence[str] = ("toxic",),
) -> dict[str, Any]:
    entries_by_key = {str(item.get("key")): item for item in memory.get("entries", []) if isinstance(item, dict)}
    scopes = memory.get("config", {}).get("scopes", {})
    tags = extract_regime_tags(row)
    matches: list[dict[str, Any]] = []
    all_matches: list[dict[str, Any]] = []
    labels = {str(label) for label in actionable_labels}

    for signal in build_shadow_expert_signals(dict(row)):
        if not signal.veto:
            continue
        first_match: dict[str, Any] | None = None
        actionable_match: dict[str, Any] | None = None
        for scope in scope_preference:
            fields = tuple(scopes.get(scope, DEFAULT_SCOPES.get(scope, ())))
            key, selected_tags = _make_key(signal.name, "veto", scope, tags, fields)
            item = entries_by_key.get(key)
            if item is None:
                continue
            match = dict(item)
            match["matched_scope"] = scope
            match["matched_tags"] = selected_tags
            all_matches.append(match)
            if first_match is None:
                first_match = match
            if str(match.get("risk_label")) in labels:
                actionable_match = match
                break
        if actionable_match is not None:
            matches.append(actionable_match)
        elif first_match is not None:
            matches.append(first_match)

    toxic_matches = [item for item in matches if item.get("risk_label") == "toxic"]
    worst_roi = min([float(item.get("shrunk_roi", 0.0)) for item in matches], default=0.0)
    return {
        "veto_matches": matches,
        "all_veto_matches": all_matches,
        "toxic_veto_count": len(toxic_matches),
        "worst_shrunk_roi": float(worst_roi),
        "has_toxic_veto": bool(toxic_matches),
    }
