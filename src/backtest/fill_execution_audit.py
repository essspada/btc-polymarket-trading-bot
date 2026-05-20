from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.backtest.regime_rule_diagnostics import build_trade_features
from src.backtest.regime_sensitivity import DEFAULT_ANCHOR_PARAMS, build_quiet_consensus_chase_rule


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _is_trade_action(value: Any) -> bool:
    return str(value or "").strip().lower() == "trade"


def _pnl(value: Any, *, filled: bool = True) -> float:
    if not filled:
        return 0.0
    return float(_to_float(value, 0.0) or 0.0)


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("market_id") or row.get("market_slug") or "")


def _bucket(value: Any, cuts: Sequence[float], labels: Sequence[str], *, missing: str = "missing") -> str:
    val = _to_float(value)
    if val is None:
        return missing
    for cut, label in zip(cuts, labels, strict=False):
        if float(val) < float(cut):
            return label
    return labels[-1] if labels else str(val)


def price_bucket(value: Any) -> str:
    return _bucket(value, (0.25, 0.50, 0.75, 0.90), ("<0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", ">=0.90"))


def fill_probability_bucket(value: Any) -> str:
    return _bucket(value, (0.35, 0.50, 0.65, 0.80), ("<0.35", "0.35-0.50", "0.50-0.65", "0.65-0.80", ">=0.80"))


def seconds_to_expiry_bucket(value: Any) -> str:
    return _bucket(value, (45, 90, 150, 240), ("<45s", "45-90s", "90-150s", "150-240s", ">=240s"))


def _decision_paper_pnl(decision: Mapping[str, Any]) -> float:
    return _pnl(decision.get("paper_sim_pnl"), filled=bool(decision.get("paper_sim_filled")))


def _decision_assume_pnl(decision: Mapping[str, Any]) -> float:
    return _pnl(decision.get("pnl_if_filled"), filled=decision.get("pnl_if_filled") is not None)


def _quiet_flags(rows: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]], *, run_name: str, fill_mode: str) -> dict[str, bool]:
    rule = build_quiet_consensus_chase_rule(DEFAULT_ANCHOR_PARAMS)
    out: dict[str, bool] = {}
    for feature in build_trade_features(rows, decisions, run_name=run_name, fill_mode=fill_mode):
        key = _row_key(feature)
        if key:
            out[key] = bool(rule.matches(feature))
    return out


def build_policy_row_audit(
    rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    run_name: str,
    policy_name: str,
    fill_mode: str = "paper_sim",
) -> list[dict[str, Any]]:
    """Compare saved runtime rows with one redecision policy row-by-row."""

    row_by_key = {_row_key(row): row for row in rows if _row_key(row)}
    quiet_by_key = _quiet_flags(rows, decisions, run_name=run_name, fill_mode=fill_mode)
    audited: list[dict[str, Any]] = []

    for decision in decisions:
        key = _row_key(decision)
        row = row_by_key.get(key, {})
        original_trade = _is_trade_action(row.get("decision_action")) or row.get("trade_id") not in {None, ""}
        original_filled = bool(row.get("trade_filled"))
        original_pnl = _pnl(row.get("trade_net_pnl"), filled=original_filled)
        policy_trade = _is_trade_action(decision.get("action"))
        policy_filled = bool(decision.get("paper_sim_filled")) if policy_trade else False
        policy_paper_pnl = _decision_paper_pnl(decision) if policy_trade else 0.0
        policy_assume_pnl = _decision_assume_pnl(decision) if policy_trade else 0.0
        selected_side = str(decision.get("selected_side") or "").strip().lower() or None
        original_side = str(row.get("predicted_side") or "").strip().lower() or None
        price = _to_float(decision.get("price"))

        audited.append(
            {
                "run": str(run_name),
                "policy": str(policy_name),
                "market_id": decision.get("market_id") or row.get("market_id"),
                "market_slug": decision.get("market_slug") or row.get("market_slug"),
                "created_at": decision.get("created_at") or row.get("created_at") or row.get("ts_utc"),
                "actual_side": decision.get("actual_side") or row.get("actual_side"),
                "original_trade": bool(original_trade),
                "original_filled": bool(original_filled),
                "original_pnl": float(original_pnl),
                "original_side": original_side,
                "original_reason": row.get("decision_reason"),
                "policy_trade": bool(policy_trade),
                "policy_filled": bool(policy_filled),
                "policy_paper_pnl": float(policy_paper_pnl),
                "policy_assume_pnl": float(policy_assume_pnl),
                "policy_side": selected_side,
                "same_side_as_original": bool(original_side in {"up", "down"} and selected_side == original_side),
                "policy_reason": decision.get("reason"),
                "policy_order_type": decision.get("order_type"),
                "policy_price": price,
                "policy_size": _to_float(decision.get("size")),
                "policy_fill_probability": _to_float(decision.get("paper_sim_fill_probability") or decision.get("fill_probability")),
                "policy_expected_roi_cash": _to_float(decision.get("expected_roi_cash")),
                "policy_breakeven_margin": _to_float(decision.get("breakeven_margin")),
                "seconds_to_expiry": _to_float(row.get("seconds_to_expiry")),
                "spot_return_bps_from_open": _to_float(row.get("spot_return_bps_from_open")),
                "spot_recent_return_1m_bps": _to_float(row.get("spot_recent_return_1m_bps")),
                "spot_recent_vol_5m_bps": _to_float(row.get("spot_recent_vol_5m_bps")),
                "quiet_consensus_chase": bool(quiet_by_key.get(key, False)),
                "price_bucket": price_bucket(price),
                "fill_probability_bucket": fill_probability_bucket(
                    decision.get("paper_sim_fill_probability") or decision.get("fill_probability")
                ),
                "seconds_to_expiry_bucket": seconds_to_expiry_bucket(row.get("seconds_to_expiry")),
            }
        )
    return audited


def summarize_audit_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    original_trade_rows = [row for row in items if row.get("original_trade")]
    policy_trade_rows = [row for row in items if row.get("policy_trade")]
    both_trade = [row for row in items if row.get("original_trade") and row.get("policy_trade")]
    policy_only = [row for row in items if row.get("policy_trade") and not row.get("original_trade")]
    original_only = [row for row in items if row.get("original_trade") and not row.get("policy_trade")]
    filled_both = [row for row in both_trade if row.get("original_filled") or row.get("policy_filled")]
    fill_agree = [row for row in both_trade if bool(row.get("original_filled")) == bool(row.get("policy_filled"))]
    fill_mismatch = [row for row in both_trade if bool(row.get("original_filled")) != bool(row.get("policy_filled"))]

    return {
        "rows": len(items),
        "original_trades": len(original_trade_rows),
        "original_filled": sum(1 for row in original_trade_rows if row.get("original_filled")),
        "original_pnl": sum(float(row.get("original_pnl", 0.0)) for row in original_trade_rows),
        "policy_trades": len(policy_trade_rows),
        "policy_filled": sum(1 for row in policy_trade_rows if row.get("policy_filled")),
        "policy_paper_pnl": sum(float(row.get("policy_paper_pnl", 0.0)) for row in policy_trade_rows),
        "policy_assume_pnl": sum(float(row.get("policy_assume_pnl", 0.0)) for row in policy_trade_rows),
        "both_trade": len(both_trade),
        "same_side_both_trade": sum(1 for row in both_trade if row.get("same_side_as_original")),
        "original_only_trades": len(original_only),
        "policy_only_trades": len(policy_only),
        "fill_agree_both_trade": len(fill_agree),
        "fill_mismatch_both_trade": len(fill_mismatch),
        "fill_agreement_rate_both_trade": float(len(fill_agree) / len(both_trade)) if both_trade else 0.0,
        "filled_either_both_trade": len(filled_both),
    }


def summarize_by_field(rows: Iterable[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field) if row.get(field) is not None else "missing")].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda item: item[0]):
        summary = summarize_audit_rows(items)
        summary[field] = key
        out.append(summary)
    return out


def compare_policy_summaries(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact pairwise deltas between policy summaries."""

    names = list(summaries)
    comparisons: list[dict[str, Any]] = []
    for left in names:
        for right in names:
            if left >= right:
                continue
            a = summaries[left]
            b = summaries[right]
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "delta_policy_trades_right_minus_left": int(b.get("policy_trades", 0)) - int(a.get("policy_trades", 0)),
                    "delta_policy_filled_right_minus_left": int(b.get("policy_filled", 0)) - int(a.get("policy_filled", 0)),
                    "delta_policy_paper_pnl_right_minus_left": float(b.get("policy_paper_pnl", 0.0)) - float(a.get("policy_paper_pnl", 0.0)),
                    "delta_policy_assume_pnl_right_minus_left": float(b.get("policy_assume_pnl", 0.0)) - float(a.get("policy_assume_pnl", 0.0)),
                }
            )
    return {"comparisons": comparisons}
