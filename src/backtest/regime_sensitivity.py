from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.backtest.regime_rule_diagnostics import RuleClause, RuleSpec, evaluate_skip_rule


@dataclass(frozen=True)
class QuietRuleParams:
    agreement_min: float
    chase_absret_min: float
    recent_flat_max: float
    vol5m_max: float


DEFAULT_AGREEMENT_VALUES = (0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 1.00)
DEFAULT_CHASE_ABSRET_VALUES = (0.50, 1.00, 1.50, 2.00, 3.00)
DEFAULT_RECENT_FLAT_VALUES = (0.25, 0.50, 0.75, 1.00, 1.50)
DEFAULT_VOL5M_VALUES = (1.50, 2.00, 2.50, 3.00, 4.00)
DEFAULT_ANCHOR_PARAMS = QuietRuleParams(
    agreement_min=0.80,
    chase_absret_min=1.00,
    recent_flat_max=0.50,
    vol5m_max=2.00,
)


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _fmt(value: float) -> str:
    return f"{float(value):g}"


def quiet_rule_name(params: QuietRuleParams) -> str:
    return (
        f"agreement>={_fmt(params.agreement_min)}"
        f" & chase_absret>={_fmt(params.chase_absret_min)}"
        f" & recent_flat<={_fmt(params.recent_flat_max)}"
        f" & vol5m<={_fmt(params.vol5m_max)}"
    )


def build_quiet_consensus_chase_rule(params: QuietRuleParams) -> RuleSpec:
    """Build an interpretable quiet-consensus-chase diagnostic skip rule."""

    return RuleSpec(
        name=quiet_rule_name(params),
        clauses=(
            RuleClause(
                f"agreement>={_fmt(params.agreement_min)}",
                lambda trade, threshold=params.agreement_min: _to_float(trade.get("agreement_ratio")) is not None
                and float(trade["agreement_ratio"]) >= float(threshold),
            ),
            RuleClause(
                f"chase_absret>={_fmt(params.chase_absret_min)}",
                lambda trade, threshold=params.chase_absret_min: bool(trade.get("is_chase"))
                and _to_float(trade.get("spot_return_bps_from_open")) is not None
                and abs(float(trade["spot_return_bps_from_open"])) >= float(threshold),
            ),
            RuleClause(
                f"recent_flat<={_fmt(params.recent_flat_max)}",
                lambda trade, threshold=params.recent_flat_max: _to_float(trade.get("spot_recent_return_1m_bps")) is not None
                and abs(float(trade["spot_recent_return_1m_bps"])) <= float(threshold),
            ),
            RuleClause(
                f"vol5m<={_fmt(params.vol5m_max)}",
                lambda trade, threshold=params.vol5m_max: _to_float(trade.get("spot_recent_vol_5m_bps")) is not None
                and float(trade["spot_recent_vol_5m_bps"]) <= float(threshold),
            ),
        ),
    )


def build_quiet_consensus_chase_grid(
    *,
    agreement_values: Sequence[float] = DEFAULT_AGREEMENT_VALUES,
    chase_absret_values: Sequence[float] = DEFAULT_CHASE_ABSRET_VALUES,
    recent_flat_values: Sequence[float] = DEFAULT_RECENT_FLAT_VALUES,
    vol5m_values: Sequence[float] = DEFAULT_VOL5M_VALUES,
) -> list[tuple[QuietRuleParams, RuleSpec]]:
    grid: list[tuple[QuietRuleParams, RuleSpec]] = []
    seen: set[QuietRuleParams] = set()
    for agreement in agreement_values:
        for chase_absret in chase_absret_values:
            for recent_flat in recent_flat_values:
                for vol5m in vol5m_values:
                    params = QuietRuleParams(
                        agreement_min=float(agreement),
                        chase_absret_min=float(chase_absret),
                        recent_flat_max=float(recent_flat),
                        vol5m_max=float(vol5m),
                    )
                    if params in seen:
                        continue
                    seen.add(params)
                    grid.append((params, build_quiet_consensus_chase_rule(params)))
    return grid


def evaluate_rule_grid(
    trades: Iterable[Mapping[str, Any]],
    grid: Sequence[tuple[QuietRuleParams, RuleSpec]],
) -> list[dict[str, Any]]:
    trades_list = list(trades)
    results: list[dict[str, Any]] = []
    for params, rule in grid:
        item = evaluate_skip_rule(trades_list, rule)
        item["params"] = asdict(params)
        results.append(item)
    return results


def target_run_item(result: Mapping[str, Any], target_run: str | None) -> Mapping[str, Any]:
    if not target_run:
        return {}
    per_run = result.get("per_run")
    if isinstance(per_run, list):
        for item in per_run:
            if item.get("run") == target_run:
                return item if isinstance(item, dict) else {}
    return {}


def robust_result(
    result: Mapping[str, Any],
    *,
    min_delta: float = 0.0,
    max_negative_delta_runs: int = 0,
    min_skipped_filled: int = 1,
    target_run: str | None = None,
    require_target_overlay_nonnegative: bool = False,
) -> bool:
    if float(result.get("delta", 0.0)) < float(min_delta):
        return False
    if int(result.get("negative_delta_runs", 0)) > int(max_negative_delta_runs):
        return False
    if int(result.get("skipped_filled", result.get("skipped_trades", 0))) < int(min_skipped_filled):
        return False
    if require_target_overlay_nonnegative and target_run:
        target = target_run_item(result, target_run)
        if float(target.get("overlay_pnl", -1e18)) < 0.0:
            return False
    return True


def rank_results_for_robustness(results: Sequence[Mapping[str, Any]], *, target_run: str | None = None) -> list[dict[str, Any]]:
    def key(result: Mapping[str, Any]) -> tuple[Any, ...]:
        target = target_run_item(result, target_run)
        target_overlay = _to_float(target.get("overlay_pnl"), -1e18) if target else -1e18
        target_delta = _to_float(target.get("delta"), 0.0) if target else 0.0
        return (
            -int(result.get("negative_delta_runs", 0)),
            float(result.get("delta", 0.0)),
            float(target_delta or 0.0),
            float(target_overlay if target_overlay is not None else -1e18),
            int(result.get("skipped_filled", result.get("skipped_trades", 0))),
            -int(result.get("skipped_trades", 0)),
        )

    return [dict(item) for item in sorted(results, key=key, reverse=True)]


def summarize_trades_by_run(trades: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"run": "", "trades": 0, "filled": 0, "wins": 0, "losses": 0, "pnl": 0.0})
    for trade in trades:
        run = str(trade.get("run") or "unknown")
        item = stats[run]
        item["run"] = run
        pnl = float(_to_float(trade.get("pnl"), 0.0) or 0.0)
        filled = bool(trade.get("filled", True))
        item["trades"] += 1
        item["filled"] += int(filled)
        item["wins"] += int(filled and pnl > 0.0)
        item["losses"] += int(filled and pnl < 0.0)
        item["pnl"] += pnl
    return [dict(item) for item in sorted(stats.values(), key=lambda x: str(x.get("run") or ""))]


def filter_trades_by_runs(trades: Iterable[Mapping[str, Any]], run_names: Iterable[str]) -> list[Mapping[str, Any]]:
    allowed = {str(name) for name in run_names}
    return [trade for trade in trades if str(trade.get("run") or "unknown") in allowed]


def _params_from_result(result: Mapping[str, Any]) -> QuietRuleParams | None:
    raw = result.get("params")
    if not isinstance(raw, dict):
        return None
    try:
        return QuietRuleParams(
            agreement_min=float(raw["agreement_min"]),
            chase_absret_min=float(raw["chase_absret_min"]),
            recent_flat_max=float(raw["recent_flat_max"]),
            vol5m_max=float(raw["vol5m_max"]),
        )
    except Exception:
        return None


def _index_distance(value: float, anchor: float, values: Sequence[float]) -> int:
    indexed = {float(item): idx for idx, item in enumerate(values)}
    if float(value) not in indexed or float(anchor) not in indexed:
        return 999
    return abs(indexed[float(value)] - indexed[float(anchor)])


def is_local_neighbor(
    params: QuietRuleParams,
    anchor: QuietRuleParams = DEFAULT_ANCHOR_PARAMS,
    *,
    agreement_values: Sequence[float] = DEFAULT_AGREEMENT_VALUES,
    chase_absret_values: Sequence[float] = DEFAULT_CHASE_ABSRET_VALUES,
    recent_flat_values: Sequence[float] = DEFAULT_RECENT_FLAT_VALUES,
    vol5m_values: Sequence[float] = DEFAULT_VOL5M_VALUES,
    max_index_distance: int = 1,
) -> bool:
    return (
        _index_distance(params.agreement_min, anchor.agreement_min, agreement_values) <= max_index_distance
        and _index_distance(params.chase_absret_min, anchor.chase_absret_min, chase_absret_values) <= max_index_distance
        and _index_distance(params.recent_flat_max, anchor.recent_flat_max, recent_flat_values) <= max_index_distance
        and _index_distance(params.vol5m_max, anchor.vol5m_max, vol5m_values) <= max_index_distance
    )


def summarize_sensitivity(
    results: Sequence[Mapping[str, Any]],
    *,
    anchor: QuietRuleParams = DEFAULT_ANCHOR_PARAMS,
    target_run: str | None = None,
    min_delta: float = 0.0,
    max_negative_delta_runs: int = 0,
    min_skipped_filled: int = 1,
    require_target_overlay_nonnegative: bool = False,
) -> dict[str, Any]:
    ranked = rank_results_for_robustness(results, target_run=target_run)
    anchor_name = quiet_rule_name(anchor)
    anchor_result = next((dict(item) for item in ranked if item.get("rule") == anchor_name), None)
    anchor_rank = next((idx + 1 for idx, item in enumerate(ranked) if item.get("rule") == anchor_name), None)

    robust = [
        item
        for item in results
        if robust_result(
            item,
            min_delta=min_delta,
            max_negative_delta_runs=max_negative_delta_runs,
            min_skipped_filled=min_skipped_filled,
            target_run=target_run,
            require_target_overlay_nonnegative=require_target_overlay_nonnegative,
        )
    ]
    local = [item for item in results if (params := _params_from_result(item)) is not None and is_local_neighbor(params, anchor)]
    local_robust = [
        item
        for item in local
        if robust_result(
            item,
            min_delta=min_delta,
            max_negative_delta_runs=max_negative_delta_runs,
            min_skipped_filled=min_skipped_filled,
            target_run=target_run,
            require_target_overlay_nonnegative=require_target_overlay_nonnegative,
        )
    ]

    return {
        "anchor_rule": anchor_name,
        "anchor_rank": anchor_rank,
        "anchor_result": anchor_result,
        "rule_count": len(results),
        "robust_rule_count": len(robust),
        "local_neighbor_count": len(local),
        "local_robust_rule_count": len(local_robust),
        "criteria": {
            "min_delta": float(min_delta),
            "max_negative_delta_runs": int(max_negative_delta_runs),
            "min_skipped_filled": int(min_skipped_filled),
            "require_target_overlay_nonnegative": bool(require_target_overlay_nonnegative),
            "target_run": target_run,
        },
    }


def train_test_validate_grid(
    trades: Iterable[Mapping[str, Any]],
    grid: Sequence[tuple[QuietRuleParams, RuleSpec]],
    *,
    train_runs: Sequence[str],
    test_runs: Sequence[str],
    target_run: str | None = None,
    top_n: int = 30,
    min_train_delta: float = 0.0,
    max_train_negative_delta_runs: int = 0,
    min_train_skipped_filled: int = 1,
) -> dict[str, Any]:
    trades_list = list(trades)
    train_trades = filter_trades_by_runs(trades_list, train_runs)
    test_trades = filter_trades_by_runs(trades_list, test_runs)
    train_results = evaluate_rule_grid(train_trades, grid)
    test_results = evaluate_rule_grid(test_trades, grid)
    test_by_rule = {str(item.get("rule")): item for item in test_results}

    eligible_train = [
        item
        for item in train_results
        if robust_result(
            item,
            min_delta=min_train_delta,
            max_negative_delta_runs=max_train_negative_delta_runs,
            min_skipped_filled=min_train_skipped_filled,
        )
    ]
    ranked_train = rank_results_for_robustness(eligible_train, target_run=None)
    selected: list[dict[str, Any]] = []
    for train_item in ranked_train[: int(top_n)]:
        rule = str(train_item.get("rule"))
        test_item = dict(test_by_rule.get(rule, {}))
        selected.append({"rule": rule, "params": train_item.get("params"), "train": dict(train_item), "test": test_item})

    return {
        "train_runs": list(train_runs),
        "test_runs": list(test_runs),
        "target_run": target_run,
        "train_trade_count": len(train_trades),
        "test_trade_count": len(test_trades),
        "eligible_train_rule_count": len(eligible_train),
        "selected_rules": selected,
        "train_sensitivity": summarize_sensitivity(
            train_results,
            target_run=None,
            min_delta=min_train_delta,
            max_negative_delta_runs=max_train_negative_delta_runs,
            min_skipped_filled=min_train_skipped_filled,
        ),
        "test_sensitivity": summarize_sensitivity(
            test_results,
            target_run=target_run,
            min_delta=0.0,
            max_negative_delta_runs=0,
            min_skipped_filled=1,
            require_target_overlay_nonnegative=bool(target_run),
        ),
    }
