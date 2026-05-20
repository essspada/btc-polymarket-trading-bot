from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _side_probability(side: str | None, p_up: float | None) -> float | None:
    if p_up is None:
        return None
    if side == "up":
        return float(p_up)
    if side == "down":
        return float(1.0 - p_up)
    return None


def _candidate_agreement(row: Mapping[str, Any], selected_side: str | None) -> dict[str, Any]:
    candidates = row.get("candidate_models")
    if selected_side not in {"up", "down"} or not isinstance(candidates, Mapping):
        return {"agreement": None, "agree": 0, "total": 0}
    agree = 0
    total = 0
    for payload in candidates.values():
        if not isinstance(payload, Mapping):
            continue
        p_up = _to_float(payload.get("p_up"))
        if p_up is None:
            continue
        side = "up" if float(p_up) >= 0.5 else "down"
        total += 1
        agree += int(side == selected_side)
    return {"agreement": float(agree / total) if total else None, "agree": int(agree), "total": int(total)}


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("market_id") or row.get("market_slug") or "")


def build_trade_features(
    rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    run_name: str,
    fill_mode: str = "assume_filled",
) -> list[dict[str, Any]]:
    row_by_key = {_row_key(row): row for row in rows if _row_key(row)}
    features: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.get("action") != "trade":
            continue
        row = row_by_key.get(_row_key(decision), {})
        selected_side = str(decision.get("selected_side") or "").strip().lower()
        if selected_side not in {"up", "down"}:
            selected_side = None

        p_up = _to_float(decision.get("p_up"))
        market_p_up = _to_float(row.get("market_p_up"))
        side_prob = _side_probability(selected_side, p_up)
        market_side_prob = _side_probability(selected_side, market_p_up)
        spot_ret = _to_float(row.get("spot_return_bps_from_open"))
        recent = _to_float(row.get("spot_recent_return_1m_bps"))
        agreement = _candidate_agreement(row, selected_side)

        chase = bool(
            (selected_side == "up" and spot_ret is not None and spot_ret > 0.0)
            or (selected_side == "down" and spot_ret is not None and spot_ret < 0.0)
        )
        recent_support = bool(
            (selected_side == "up" and recent is not None and recent > 0.0)
            or (selected_side == "down" and recent is not None and recent < 0.0)
        )
        recent_oppose = bool(
            (selected_side == "up" and recent is not None and recent < 0.0)
            or (selected_side == "down" and recent is not None and recent > 0.0)
        )

        fill_mode_value = str(fill_mode or "assume_filled")
        if fill_mode_value == "paper_sim":
            filled = bool(decision.get("paper_sim_filled"))
            pnl = float(_to_float(decision.get("paper_sim_pnl"), 0.0) or 0.0) if filled else 0.0
            win = bool(filled and pnl > 0.0)
        else:
            filled = decision.get("pnl_if_filled") is not None
            pnl = float(_to_float(decision.get("pnl_if_filled"), 0.0) or 0.0) if filled else 0.0
            win = bool(decision.get("win_if_filled")) if filled else False

        features.append(
            {
                "run": str(run_name),
                "market_id": decision.get("market_id"),
                "market_slug": decision.get("market_slug"),
                "created_at": decision.get("created_at"),
                "selected_side": selected_side,
                "actual_side": decision.get("actual_side"),
                "fill_mode": fill_mode_value,
                "filled": bool(filled),
                "pnl": float(pnl),
                "win": bool(win),
                "price": _to_float(decision.get("price")),
                "p_up": p_up,
                "side_probability": side_prob,
                "market_p_up": market_p_up,
                "market_side_probability": market_side_prob,
                "model_market_gap": None if side_prob is None or market_side_prob is None else float(side_prob - market_side_prob),
                "expected_roi_cash": _to_float(decision.get("expected_roi_cash")),
                "breakeven_margin": _to_float(decision.get("breakeven_margin")),
                "seconds_to_expiry": _to_float(row.get("seconds_to_expiry")),
                "spot_return_bps_from_open": spot_ret,
                "spot_recent_return_1m_bps": recent,
                "spot_recent_vol_5m_bps": _to_float(row.get("spot_recent_vol_5m_bps")),
                "agreement_ratio": agreement["agreement"],
                "agreement_count": agreement["agree"],
                "agreement_total": agreement["total"],
                "is_chase": chase,
                "recent_supports_side": recent_support,
                "recent_opposes_side": recent_oppose,
            }
        )
    return features


@dataclass(frozen=True)
class RuleClause:
    name: str
    predicate: Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class RuleSpec:
    name: str
    clauses: Sequence[RuleClause]

    def matches(self, trade: Mapping[str, Any]) -> bool:
        return all(clause.predicate(trade) for clause in self.clauses)


def _ge(field: str, threshold: float, name: str | None = None) -> RuleClause:
    label = name or f"{field}>={threshold:g}"
    return RuleClause(label, lambda trade, field=field, threshold=threshold: _to_float(trade.get(field)) is not None and float(trade[field]) >= threshold)


def _le(field: str, threshold: float, name: str | None = None) -> RuleClause:
    label = name or f"{field}<={threshold:g}"
    return RuleClause(label, lambda trade, field=field, threshold=threshold: _to_float(trade.get(field)) is not None and float(trade[field]) <= threshold)


def _abs_ge(field: str, threshold: float, name: str | None = None) -> RuleClause:
    label = name or f"abs({field})>={threshold:g}"
    return RuleClause(label, lambda trade, field=field, threshold=threshold: _to_float(trade.get(field)) is not None and abs(float(trade[field])) >= threshold)


def _abs_le(field: str, threshold: float, name: str | None = None) -> RuleClause:
    label = name or f"abs({field})<={threshold:g}"
    return RuleClause(label, lambda trade, field=field, threshold=threshold: _to_float(trade.get(field)) is not None and abs(float(trade[field])) <= threshold)


def _bool(field: str, name: str | None = None) -> RuleClause:
    return RuleClause(name or str(field), lambda trade, field=field: bool(trade.get(field)))


def _rule(*clauses: RuleClause) -> RuleSpec:
    name = " & ".join(clause.name for clause in clauses)
    return RuleSpec(name=name, clauses=clauses)


def build_phase7_rule_specs() -> list[RuleSpec]:
    """Return a bounded set of interpretable Phase7 regime rules.

    These are diagnostic skip rules for offline replay, not runtime trading rules.
    """

    agreement = [_ge("agreement_ratio", x, f"agreement>={x:g}") for x in (0.80, 0.85, 0.88)]
    low_vol = [_le("spot_recent_vol_5m_bps", x, f"vol5m<={x:g}") for x in (2.0, 3.0)]
    high_roi = [_ge("expected_roi_cash", x, f"roi>={x:g}") for x in (0.04, 0.06, 0.08, 0.10)]
    high_gap = [_ge("model_market_gap", x, f"model_market_gap>={x:g}") for x in (0.05, 0.08, 0.10, 0.15)]
    chase_move = [
        RuleClause(
            f"chase_absret>={x:g}",
            lambda trade, x=x: bool(trade.get("is_chase"))
            and _to_float(trade.get("spot_return_bps_from_open")) is not None
            and abs(float(trade["spot_return_bps_from_open"])) >= x,
        )
        for x in (1.0, 2.0, 3.0)
    ]
    recent_flat = [_abs_le("spot_recent_return_1m_bps", x, f"recent_flat<={x:g}") for x in (0.25, 0.5, 1.0)]
    expensive = [_ge("price", x, f"price>={x:g}") for x in (0.55, 0.65, 0.75)]
    recent_oppose = [
        RuleClause(
            f"recent_oppose_abs>={x:g}",
            lambda trade, x=x: bool(trade.get("recent_opposes_side"))
            and _to_float(trade.get("spot_recent_return_1m_bps")) is not None
            and abs(float(trade["spot_recent_return_1m_bps"])) >= x,
        )
        for x in (0.0, 0.25, 0.5)
    ]

    specs: dict[str, RuleSpec] = {}
    for agree in agreement:
        for vol in low_vol:
            for roi in high_roi:
                specs.setdefault(_rule(roi, agree, vol).name, _rule(roi, agree, vol))
            for gap in high_gap:
                specs.setdefault(_rule(gap, agree, vol).name, _rule(gap, agree, vol))
            for roi in high_roi:
                for gap in high_gap:
                    specs.setdefault(_rule(roi, gap, agree, vol).name, _rule(roi, gap, agree, vol))
            for chase in chase_move:
                for flat in recent_flat:
                    specs.setdefault(_rule(agree, chase, flat, vol).name, _rule(agree, chase, flat, vol))
                for price in expensive:
                    specs.setdefault(_rule(agree, chase, price, vol).name, _rule(agree, chase, price, vol))
                for oppose in recent_oppose:
                    specs.setdefault(_rule(agree, chase, oppose, vol).name, _rule(agree, chase, oppose, vol))
    for vol in low_vol:
        for chase in chase_move:
            for flat in recent_flat:
                for price in expensive:
                    specs.setdefault(_rule(chase, flat, price, vol).name, _rule(chase, flat, price, vol))
    return list(specs.values())


def evaluate_skip_rule(trades: Iterable[Mapping[str, Any]], rule: RuleSpec) -> dict[str, Any]:
    base_pnl: dict[str, float] = defaultdict(float)
    base_trades: dict[str, int] = defaultdict(int)
    base_filled: dict[str, int] = defaultdict(int)
    skipped_pnl: dict[str, float] = defaultdict(float)
    skipped_trades: dict[str, int] = defaultdict(int)
    skipped_filled: dict[str, int] = defaultdict(int)
    skipped_losses: dict[str, int] = defaultdict(int)
    skipped_wins: dict[str, int] = defaultdict(int)
    saved_losses: dict[str, float] = defaultdict(float)
    missed_profit: dict[str, float] = defaultdict(float)

    for trade in trades:
        run = str(trade.get("run") or "unknown")
        pnl = float(_to_float(trade.get("pnl"), 0.0) or 0.0)
        filled = bool(trade.get("filled", True))
        base_pnl[run] += pnl
        base_trades[run] += 1
        base_filled[run] += int(filled)
        if rule.matches(trade):
            skipped_pnl[run] += pnl
            skipped_trades[run] += 1
            skipped_filled[run] += int(filled)
            if pnl < 0.0:
                skipped_losses[run] += 1
                saved_losses[run] += -pnl
            elif pnl > 0.0:
                skipped_wins[run] += 1
                missed_profit[run] += pnl

    per_run: list[dict[str, Any]] = []
    for run in sorted(base_pnl):
        overlay = float(base_pnl[run] - skipped_pnl.get(run, 0.0))
        delta = float(overlay - base_pnl[run])
        per_run.append(
            {
                "run": run,
                "base_pnl": float(base_pnl[run]),
                "overlay_pnl": overlay,
                "delta": delta,
                "base_trades": int(base_trades[run]),
                "base_filled": int(base_filled[run]),
                "skipped_trades": int(skipped_trades.get(run, 0)),
                "skipped_filled": int(skipped_filled.get(run, 0)),
                "skipped_pnl": float(skipped_pnl.get(run, 0.0)),
                "skipped_losses": int(skipped_losses.get(run, 0)),
                "skipped_wins": int(skipped_wins.get(run, 0)),
                "saved_losses": float(saved_losses.get(run, 0.0)),
                "missed_profit": float(missed_profit.get(run, 0.0)),
            }
        )

    total_base = sum(item["base_pnl"] for item in per_run)
    total_overlay = sum(item["overlay_pnl"] for item in per_run)
    return {
        "rule": rule.name,
        "base_pnl": float(total_base),
        "overlay_pnl": float(total_overlay),
        "delta": float(total_overlay - total_base),
        "negative_delta_runs": sum(1 for item in per_run if float(item["delta"]) < -1e-9),
        "filled_trades": sum(int(item["base_filled"]) for item in per_run),
        "skipped_trades": sum(int(item["skipped_trades"]) for item in per_run),
        "skipped_filled": sum(int(item["skipped_filled"]) for item in per_run),
        "saved_losses": sum(float(item["saved_losses"]) for item in per_run),
        "missed_profit": sum(float(item["missed_profit"]) for item in per_run),
        "per_run": per_run,
    }


def rank_rule_results(results: Sequence[Mapping[str, Any]], *, target_run: str | None = None) -> list[dict[str, Any]]:
    def target_item(result: Mapping[str, Any]) -> Mapping[str, Any]:
        per_run = result.get("per_run")
        if isinstance(per_run, list):
            for item in per_run:
                if target_run is not None and item.get("run") == target_run:
                    return item
        return {}

    def key(result: Mapping[str, Any]) -> tuple[Any, ...]:
        target = target_item(result)
        target_overlay = _to_float(target.get("overlay_pnl"), -1e9) or -1e9
        target_delta = _to_float(target.get("delta"), 0.0) or 0.0
        return (
            target_overlay >= 0.0,
            -int(result.get("negative_delta_runs", 0)),
            float(result.get("delta", 0.0)),
            target_delta,
            -int(result.get("skipped_trades", 0)),
        )

    return [dict(item) for item in sorted(results, key=key, reverse=True)]
