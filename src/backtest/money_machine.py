from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class StrategySpec:
    name: str
    source_key: str = "spot_logistic_online"
    min_stage_delay: int | None = 240
    min_edge: float = 0.03
    max_spread: float | None = 0.02
    min_price: float = 0.05
    max_price: float = 0.95
    min_confidence: float | None = None
    max_confidence: float | None = None
    reverse: bool = False
    confirm_key: str | None = None
    confirm_min_edge: float = 0.03
    confirm_same_side: bool = True
    empirical_confirm: bool = False
    empirical_min_edge: float = 0.03
    empirical_warmup_rows: int = 120
    entry_price: str = "bid"


@dataclass(frozen=True)
class SizingSpec:
    name: str
    mode: str = "fractional_kelly"
    risk_fraction: float = 0.08
    max_trade_usd: float = 150.0
    min_trade_usd: float = 5.0
    kelly_multiplier: float = 0.25


@dataclass(frozen=True)
class TradeDecision:
    strategy_name: str
    source_key: str
    side: str
    price: float
    p_side: float
    edge: float
    confidence: float
    spread: float
    created_at: str
    actual_side: str
    confirm_edge: float | None = None
    empirical_edge: float | None = None
    stage_delay: int | None = None


DEFAULT_BANKROLLS = (100.0, 200.0, 500.0, 750.0, 1000.0)


DEFAULT_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="runtime_like_wide",
        source_key="spot_logistic_online",
        min_stage_delay=None,
        min_edge=0.004,
        max_spread=0.25,
        min_price=0.02,
        max_price=0.98,
    ),
    StrategySpec(
        name="pup_240_edge005",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.05,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
    ),
    StrategySpec(
        name="pup_240_edge015",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.15,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
    ),
    StrategySpec(
        name="proxy_confirmed_pup_edge003",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        confirm_key="proxy_logistic_market_blend",
        confirm_min_edge=0.03,
    ),
    StrategySpec(
        name="proxy_confirmed_pup_edge005",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.05,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        confirm_key="proxy_logistic_market_blend",
        confirm_min_edge=0.03,
    ),
    StrategySpec(
        name="proxy_alone_edge003",
        source_key="proxy_logistic_market_blend",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
    ),
    StrategySpec(
        name="pup_mid_conf_edge003",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        min_confidence=0.55,
        max_confidence=0.82,
    ),
    StrategySpec(
        name="pup_avoid_extreme_conf_edge003",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        max_confidence=0.90,
    ),
    StrategySpec(
        name="reverse_extreme_conf090",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        min_confidence=0.90,
        reverse=True,
    ),
    StrategySpec(
        name="empirical_confirmed_pup_edge003",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        empirical_confirm=True,
        empirical_min_edge=0.03,
    ),
    StrategySpec(
        name="proxy_and_empirical_confirmed",
        source_key="spot_logistic_online",
        min_stage_delay=240,
        min_edge=0.03,
        max_spread=0.02,
        min_price=0.05,
        max_price=0.95,
        confirm_key="proxy_logistic_market_blend",
        confirm_min_edge=0.03,
        empirical_confirm=True,
        empirical_min_edge=0.02,
    ),
)


DEFAULT_SIZINGS: tuple[SizingSpec, ...] = (
    SizingSpec(name="edge_scaled_12pct_cap150", mode="edge_scaled", risk_fraction=0.12, max_trade_usd=150.0, min_trade_usd=5.0),
    SizingSpec(name="kelly025_8pct_cap150", mode="fractional_kelly", risk_fraction=0.08, max_trade_usd=150.0, min_trade_usd=5.0, kelly_multiplier=0.25),
    SizingSpec(name="kelly035_12pct_cap150", mode="fractional_kelly", risk_fraction=0.12, max_trade_usd=150.0, min_trade_usd=5.0, kelly_multiplier=0.35),
    SizingSpec(name="flat_8pct_cap150", mode="flat_fraction", risk_fraction=0.08, max_trade_usd=150.0, min_trade_usd=5.0),
)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return float(out)


def clamp_prob(value: Any) -> float | None:
    val = safe_float(value)
    if val is None:
        return None
    return float(max(1e-6, min(1.0 - 1e-6, val)))


def logit(value: Any) -> float:
    p = clamp_prob(value)
    if p is None:
        p = 0.5
    return float(math.log(p / (1.0 - p)))


def parse_dt(value: Any) -> datetime:
    raw = str(value or "")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return datetime.fromtimestamp(0, tz=UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def load_resolved_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            actual = str(row.get("actual_side") or "").strip().lower()
            if actual not in {"up", "down"}:
                continue
            row = dict(row)
            row["actual_side"] = actual
            rows.append(row)
    rows.sort(key=lambda item: str(item.get("created_at") or item.get("ts_utc") or ""))
    return rows


def candidate_p_up(row: Mapping[str, Any], key: str) -> float | None:
    key = str(key).strip()
    if not key:
        return None
    if key in {"p_up", "top_level", "log180_then_log240"}:
        direct = clamp_prob(row.get("p_up"))
        if direct is not None:
            return direct
    direct_keys = [key]
    if not key.endswith("_p_up"):
        direct_keys.append(f"{key}_p_up")
    for direct_key in direct_keys:
        if direct_key in row:
            prob = clamp_prob(row.get(direct_key))
            if prob is not None:
                return prob
    nested = row.get("candidate_models")
    if isinstance(nested, Mapping):
        item = nested.get(key)
        if isinstance(item, Mapping):
            for nested_key in ("p_up", "p_up_calibrated", "p_up_pre_calibration", "p_up_raw"):
                prob = clamp_prob(item.get(nested_key))
                if prob is not None:
                    return prob
    return None


def side_price(row: Mapping[str, Any], side: str, entry_price: str = "bid") -> float | None:
    entry = str(entry_price).strip().lower()
    if entry not in {"bid", "ask"}:
        raise ValueError(f"entry_price must be 'bid' or 'ask', got {entry_price!r}")
    key = f"{side}_best_{entry}"
    price = safe_float(row.get(key))
    if price is None or price <= 0.0 or price >= 1.0:
        return None
    return float(price)


def side_spread(row: Mapping[str, Any], side: str) -> float | None:
    key = "up_spread" if side == "up" else "down_spread"
    spread = safe_float(row.get(key))
    if spread is None:
        return None
    return float(max(0.0, spread))


def _row_fee_rate(row: Mapping[str, Any], side: str, fallback_fee_rate: float, fee_rate_source: str) -> float:
    source = str(fee_rate_source).strip().lower()
    if source == "config":
        return max(0.0, float(fallback_fee_rate))
    if source != "row":
        raise ValueError(f"fee_rate_source must be 'config' or 'row', got {fee_rate_source!r}")

    for key in (f"{side}_taker_fee_bps", "selected_taker_fee_bps", "taker_fee_bps"):
        bps = safe_float(row.get(key))
        if bps is not None:
            return max(0.0, float(bps) / 10_000.0)
    return max(0.0, float(fallback_fee_rate))


def selected_side_from_prob(p_up: float, *, reverse: bool = False) -> str:
    side = "up" if p_up >= 0.5 else "down"
    if reverse:
        return "down" if side == "up" else "up"
    return side


def p_side_from_prob(p_up: float, side: str, *, reverse: bool = False) -> float:
    if reverse:
        return float(max(p_up, 1.0 - p_up))
    return float(p_up if side == "up" else 1.0 - p_up)


def row_stage_delay(row: Mapping[str, Any]) -> int | None:
    raw = row.get("timing_policy_stage_delay")
    if raw is None:
        raw = row.get("spot_timing_policy_stage_delay")
    val = safe_float(raw)
    if val is None:
        return None
    return int(val)


def _passes_stage(row: Mapping[str, Any], spec: StrategySpec) -> bool:
    if spec.min_stage_delay is None:
        return True
    stage = row_stage_delay(row)
    if stage is not None:
        return int(stage) >= int(spec.min_stage_delay)
    sec = safe_float(row.get("seconds_from_window_start"))
    return bool(sec is not None and sec >= float(spec.min_stage_delay))


def select_trade(
    row: Mapping[str, Any],
    spec: StrategySpec,
    *,
    empirical_p_up: float | None = None,
) -> TradeDecision | None:
    if not _passes_stage(row, spec):
        return None
    actual = str(row.get("actual_side") or "").strip().lower()
    if actual not in {"up", "down"}:
        return None
    p_up = candidate_p_up(row, spec.source_key)
    if p_up is None:
        return None
    confidence = float(max(p_up, 1.0 - p_up))
    if spec.min_confidence is not None and confidence < float(spec.min_confidence):
        return None
    if spec.max_confidence is not None and confidence > float(spec.max_confidence):
        return None

    side = selected_side_from_prob(float(p_up), reverse=bool(spec.reverse))
    price = side_price(row, side, spec.entry_price)
    spread = side_spread(row, side)
    if price is None or spread is None:
        return None
    if price < float(spec.min_price) or price > float(spec.max_price):
        return None
    if spec.max_spread is not None and spread > float(spec.max_spread):
        return None

    p_side = p_side_from_prob(float(p_up), side, reverse=bool(spec.reverse))
    edge = float(p_side - price)
    if edge < float(spec.min_edge):
        return None

    confirm_edge: float | None = None
    if spec.confirm_key:
        confirm_p_up = candidate_p_up(row, spec.confirm_key)
        if confirm_p_up is None:
            return None
        confirm_side = selected_side_from_prob(float(confirm_p_up), reverse=False)
        if spec.confirm_same_side and confirm_side != side:
            return None
        confirm_p_side = p_side_from_prob(float(confirm_p_up), side, reverse=False)
        confirm_edge = float(confirm_p_side - price)
        if confirm_edge < float(spec.confirm_min_edge):
            return None

    empirical_edge: float | None = None
    if spec.empirical_confirm:
        if empirical_p_up is None:
            return None
        emp_side = selected_side_from_prob(float(empirical_p_up), reverse=False)
        if emp_side != side:
            return None
        emp_p_side = p_side_from_prob(float(empirical_p_up), side, reverse=False)
        empirical_edge = float(emp_p_side - price)
        if empirical_edge < float(spec.empirical_min_edge):
            return None

    return TradeDecision(
        strategy_name=spec.name,
        source_key=spec.source_key,
        side=side,
        price=float(price),
        p_side=float(p_side),
        edge=float(edge),
        confidence=float(confidence),
        spread=float(spread),
        created_at=str(row.get("created_at") or row.get("ts_utc") or ""),
        actual_side=actual,
        confirm_edge=confirm_edge,
        empirical_edge=empirical_edge,
        stage_delay=row_stage_delay(row),
    )


def feature_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    market_p = candidate_p_up(row, "market_p_up") or 0.5
    base_p = candidate_p_up(row, "spot_logistic_online") or candidate_p_up(row, "p_up") or 0.5
    return (
        (safe_float(row.get("spot_return_bps_from_open"), 0.0) or 0.0) / 8.0,
        (safe_float(row.get("spot_recent_return_1m_bps"), 0.0) or 0.0) / 4.0,
        (safe_float(row.get("spot_recent_vol_5m_bps"), 0.0) or 0.0) / 8.0,
        logit(market_p) / 4.0,
        logit(base_p) / 5.0,
        (safe_float(row.get("up_top5_imbalance"), 0.0) or 0.0),
        (safe_float(row.get("down_top5_imbalance"), 0.0) or 0.0),
        (safe_float(row.get("seconds_to_expiry"), 60.0) or 60.0) / 300.0,
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)))


def empirical_p_up_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    warmup_rows: int = 120,
    nearest_k: int = 120,
    prior_weight: float = 8.0,
) -> list[float | None]:
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    out: list[float | None] = []
    for row in rows:
        current_feature = feature_vector(row)
        if len(labels) < int(warmup_rows):
            out.append(None)
        else:
            distances = [(_distance(current_feature, prev_feature), idx) for idx, prev_feature in enumerate(features)]
            distances.sort(key=lambda item: item[0])
            selected = distances[: max(1, min(int(nearest_k), len(distances)))]
            weighted_sum = 0.5 * float(prior_weight)
            weight_total = float(prior_weight)
            for dist, idx in selected:
                weight = 1.0 / (1.0 + float(dist))
                weighted_sum += weight * float(labels[idx])
                weight_total += weight
            out.append(float(weighted_sum / max(1e-9, weight_total)))
        actual = str(row.get("actual_side") or "").strip().lower()
        features.append(current_feature)
        labels.append(1 if actual == "up" else 0)
    return out


def cash_for_trade(balance: float, decision: TradeDecision, strategy: StrategySpec, sizing: SizingSpec) -> float:
    balance_f = max(0.0, float(balance))
    if balance_f <= 0.0:
        return 0.0
    cap = min(balance_f, balance_f * max(0.0, float(sizing.risk_fraction)), max(0.0, float(sizing.max_trade_usd)))
    if cap <= 0.0:
        return 0.0
    mode = str(sizing.mode).strip().lower()
    if mode == "flat_fraction":
        cash = cap
    elif mode == "edge_scaled":
        denom = max(1e-6, 5.0 * max(1e-6, float(strategy.min_edge)))
        strength = max(0.0, min(1.0, (float(decision.edge) - float(strategy.min_edge)) / denom))
        cash = cap * strength
    elif mode == "fractional_kelly":
        # Binary Kelly for buying a $1-settled share at price q: f* = (p-q)/(1-q).
        # We always cap it because model probabilities are not trustworthy enough for full Kelly.
        kelly_fraction = max(0.0, float(decision.edge) / max(1e-6, 1.0 - float(decision.price)))
        cash = balance_f * min(max(0.0, float(sizing.risk_fraction)), kelly_fraction * max(0.0, float(sizing.kelly_multiplier)))
        cash = min(cash, cap)
    else:
        raise ValueError(f"unknown sizing mode: {sizing.mode}")
    if cash < float(sizing.min_trade_usd):
        return 0.0
    return float(min(cash, balance_f))


def pnl_for_trade(
    cash: float,
    price: float,
    side: str,
    actual_side: str,
    fee_rate: float = 0.0,
    slippage_bps: float = 0.0,
) -> float:
    cash_f = float(cash)
    if cash_f <= 0.0:
        return 0.0
    price_f = max(1e-6, min(1.0 - 1e-6, float(price)))
    size = cash_f / price_f
    # Polymarket V2 binary fee: size * feeRate * price * (1 - price)
    fee = size * float(fee_rate) * price_f * (1.0 - price_f) if fee_rate > 0 else 0.0
    slip = size * price_f * (float(slippage_bps) / 10_000.0) if slippage_bps > 0 else 0.0
    if str(side).lower() == str(actual_side).lower():
        return float(size * (1.0 - price_f) - fee - slip)
    return float(-cash_f - fee - slip)


def _bucket(value: float, edges: Sequence[float]) -> str:
    prev = None
    for edge in edges:
        if value < float(edge):
            if prev is None:
                return f"<{edge:g}"
            return f"{prev:g}-{edge:g}"
        prev = float(edge)
    return f">={edges[-1]:g}"


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(mean(values)), "min": float(min(values)), "max": float(max(values))}


def simulate_bankroll(
    rows: Sequence[Mapping[str, Any]],
    strategy: StrategySpec,
    sizing: SizingSpec,
    *,
    initial_balance: float,
    empirical_probs: Sequence[float | None] | None = None,
    win_fill_factor: float = 1.0,
    loss_fill_factor: float = 1.0,
    fee_rate: float = 0.0,
    fee_rate_source: str = "config",
    slippage_bps: float = 0.0,
    include_trades: bool = False,
) -> dict[str, Any]:
    balance = float(initial_balance)
    peak = float(balance)
    max_drawdown_pct = 0.0
    min_balance = float(balance)
    trades = 0
    wins = 0
    selected = 0
    no_cash = 0
    skipped = 0
    pnl_total = 0.0
    profit_sum = 0.0
    loss_sum = 0.0
    cash_values: list[float] = []
    edge_values: list[float] = []
    trade_rows: list[dict[str, Any]] = []
    current_loss_streak = 0
    max_loss_streak = 0

    for idx, row in enumerate(rows):
        emp = empirical_probs[idx] if empirical_probs is not None and idx < len(empirical_probs) else None
        decision = select_trade(row, strategy, empirical_p_up=emp)
        if decision is None:
            skipped += 1
            continue
        selected += 1
        cash = cash_for_trade(balance, decision, strategy, sizing)
        if cash <= 0.0:
            no_cash += 1
            continue
        effective_fee_rate = _row_fee_rate(row, decision.side, fee_rate, fee_rate_source)
        raw_pnl = pnl_for_trade(
            cash,
            decision.price,
            decision.side,
            decision.actual_side,
            fee_rate=effective_fee_rate,
            slippage_bps=slippage_bps,
        )
        filled_factor = float(win_fill_factor if raw_pnl >= 0.0 else loss_fill_factor)
        pnl = float(raw_pnl * max(0.0, min(1.0, filled_factor)))
        before = float(balance)
        balance = float(balance + pnl)
        peak = max(float(peak), float(balance))
        min_balance = min(float(min_balance), float(balance))
        drawdown = max(0.0, (float(peak) - float(balance)) / max(1e-9, float(peak)))
        max_drawdown_pct = max(float(max_drawdown_pct), float(drawdown))
        trades += 1
        is_win = raw_pnl >= 0.0
        wins += int(is_win)
        if is_win:
            current_loss_streak = 0
            profit_sum += pnl
        else:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            loss_sum += abs(pnl)
        pnl_total += pnl
        cash_values.append(float(cash))
        edge_values.append(float(decision.edge))
        if include_trades:
            trade_rows.append(
                {
                    "idx": idx,
                    "created_at": decision.created_at,
                    "side": decision.side,
                    "actual_side": decision.actual_side,
                    "win": bool(is_win),
                    "price": decision.price,
                    "p_side": decision.p_side,
                    "edge": decision.edge,
                    "confidence": decision.confidence,
                    "cash": float(cash),
                    "fee_rate": float(effective_fee_rate),
                    "pnl": float(pnl),
                    "balance_before": before,
                    "balance_after": float(balance),
                    "stage_delay": decision.stage_delay,
                    "confirm_edge": decision.confirm_edge,
                    "empirical_edge": decision.empirical_edge,
                    "market_id": row.get("market_id"),
                    "market_slug": row.get("market_slug"),
                }
            )

    return {
        "strategy": strategy.name,
        "sizing": sizing.name,
        "initial_balance": float(initial_balance),
        "final_balance": float(balance),
        "pnl": float(pnl_total),
        "return_pct": float((balance / initial_balance - 1.0) * 100.0) if initial_balance else 0.0,
        "return_multiple": float(balance / initial_balance) if initial_balance else 0.0,
        "selected_signals": int(selected),
        "trades": int(trades),
        "wins": int(wins),
        "losses": int(trades - wins),
        "win_rate": float(wins / trades) if trades else 0.0,
        "skipped_no_signal": int(skipped),
        "skipped_no_cash_or_min_trade": int(no_cash),
        "max_drawdown_pct": float(max_drawdown_pct * 100.0),
        "min_balance": float(min_balance),
        "peak_balance": float(peak),
        "max_loss_streak": int(max_loss_streak),
        "profit_factor": float(profit_sum / loss_sum) if loss_sum > 0.0 else (999.0 if profit_sum > 0 else 0.0),
        "cash_stats": _summary(cash_values),
        "edge_stats": _summary(edge_values),
        "win_fill_factor": float(win_fill_factor),
        "loss_fill_factor": float(loss_fill_factor),
        "fee_rate_source": str(fee_rate_source),
        "trades_log": trade_rows if include_trades else None,
    }


def fixed_cash_metrics(
    rows: Sequence[Mapping[str, Any]],
    strategy: StrategySpec,
    *,
    empirical_probs: Sequence[float | None] | None = None,
    cash: float = 100.0,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        emp = empirical_probs[idx] if empirical_probs is not None and idx < len(empirical_probs) else None
        decision = select_trade(row, strategy, empirical_p_up=emp)
        if decision is None:
            continue
        pnl = pnl_for_trade(float(cash), decision.price, decision.side, decision.actual_side)
        dt = parse_dt(decision.created_at)
        trades.append(
            {
                "idx": idx,
                "created_at": decision.created_at,
                "window_12h": dt.replace(hour=(dt.hour // 12) * 12, minute=0, second=0, microsecond=0).isoformat(),
                "side": decision.side,
                "actual_side": decision.actual_side,
                "win": pnl >= 0.0,
                "price": decision.price,
                "edge": decision.edge,
                "confidence": decision.confidence,
                "spread": decision.spread,
                "pnl": float(pnl),
                "stage_delay": decision.stage_delay,
                "confirm_edge": decision.confirm_edge,
                "empirical_edge": decision.empirical_edge,
            }
        )
    pnl_sum = sum(float(item["pnl"]) for item in trades)
    wins = sum(1 for item in trades if bool(item["win"]))
    by_12h: dict[str, float] = defaultdict(float)
    for item in trades:
        by_12h[str(item["window_12h"])] += float(item["pnl"])
    adverse: dict[str, float] = {}
    for gap in (0.10, 0.15, 0.20):
        total = 0.0
        for item in trades:
            factor = 1.0 - gap if bool(item["win"]) else 1.0
            total += float(item["pnl"]) * factor
        adverse[f"adverse_{int(gap * 100)}pp"] = float(total)
    return {
        "trades": len(trades),
        "wins": int(wins),
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "pnl": float(pnl_sum),
        "avg_pnl": float(pnl_sum / len(trades)) if trades else 0.0,
        "worst_trade": float(min((item["pnl"] for item in trades), default=0.0)),
        "best_trade": float(max((item["pnl"] for item in trades), default=0.0)),
        "count_12h": int(len(by_12h)),
        "positive_12h": int(sum(1 for value in by_12h.values() if value > 0.0)),
        "worst_12h": float(min(by_12h.values(), default=0.0)),
        "best_12h": float(max(by_12h.values(), default=0.0)),
        **adverse,
    }


def diagnostic_buckets(
    rows: Sequence[Mapping[str, Any]],
    strategy: StrategySpec,
    *,
    empirical_probs: Sequence[float | None] | None = None,
    cash: float = 100.0,
) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = {
        "confidence": Counter(),
        "edge": Counter(),
        "price": Counter(),
        "side": Counter(),
        "hour_utc": Counter(),
    }
    sums: dict[str, defaultdict[str, float]] = {name: defaultdict(float) for name in buckets}
    wins: dict[str, Counter[str]] = {name: Counter() for name in buckets}
    for idx, row in enumerate(rows):
        emp = empirical_probs[idx] if empirical_probs is not None and idx < len(empirical_probs) else None
        decision = select_trade(row, strategy, empirical_p_up=emp)
        if decision is None:
            continue
        pnl = pnl_for_trade(float(cash), decision.price, decision.side, decision.actual_side)
        dt = parse_dt(decision.created_at)
        keys = {
            "confidence": _bucket(decision.confidence, (0.60, 0.70, 0.80, 0.90, 0.97)),
            "edge": _bucket(decision.edge, (0.03, 0.05, 0.08, 0.12, 0.20, 0.35)),
            "price": _bucket(decision.price, (0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)),
            "side": decision.side,
            "hour_utc": f"{dt.hour:02d}",
        }
        for name, key in keys.items():
            buckets[name][key] += 1
            sums[name][key] += float(pnl)
            wins[name][key] += int(pnl >= 0.0)
    out: dict[str, Any] = {}
    for name, counter in buckets.items():
        out[name] = [
            {
                "bucket": key,
                "trades": int(count),
                "win_rate": float(wins[name][key] / count) if count else 0.0,
                "pnl": float(sums[name][key]),
            }
            for key, count in sorted(counter.items())
        ]
    return out


def segments_12h(
    rows: Sequence[Mapping[str, Any]],
    strategy: StrategySpec,
    *,
    empirical_probs: Sequence[float | None] | None = None,
    cash: float = 100.0,
) -> list[dict[str, Any]]:
    segments: dict[str, dict[str, Any]] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for idx, row in enumerate(rows):
        emp = empirical_probs[idx] if empirical_probs is not None and idx < len(empirical_probs) else None
        decision = select_trade(row, strategy, empirical_p_up=emp)
        if decision is None:
            continue
        pnl = pnl_for_trade(float(cash), decision.price, decision.side, decision.actual_side)
        dt = parse_dt(decision.created_at)
        key = dt.replace(hour=(dt.hour // 12) * 12, minute=0, second=0, microsecond=0).isoformat()
        segments[key]["trades"] += 1
        segments[key]["wins"] += int(pnl >= 0.0)
        segments[key]["pnl"] += float(pnl)
    out: list[dict[str, Any]] = []
    for key, value in sorted(segments.items()):
        trades = int(value["trades"])
        wins = int(value["wins"])
        out.append(
            {
                "segment_start_utc": key,
                "trades": trades,
                "wins": wins,
                "win_rate": float(wins / trades) if trades else 0.0,
                "pnl": float(value["pnl"]),
            }
        )
    return out


def evaluate_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategies: Sequence[StrategySpec] = DEFAULT_STRATEGIES,
    sizings: Sequence[SizingSpec] = DEFAULT_SIZINGS,
    bankrolls: Sequence[float] = DEFAULT_BANKROLLS,
    empirical_probs: Sequence[float | None] | None = None,
) -> dict[str, Any]:
    fixed: list[dict[str, Any]] = []
    bankroll: list[dict[str, Any]] = []
    for strategy in strategies:
        fixed.append({"strategy": strategy.name, **fixed_cash_metrics(rows, strategy, empirical_probs=empirical_probs)})
        for sizing in sizings:
            for initial in bankrolls:
                bankroll.append(
                    simulate_bankroll(
                        rows,
                        strategy,
                        sizing,
                        initial_balance=float(initial),
                        empirical_probs=empirical_probs,
                    )
                )
    return {"fixed_cash_100": fixed, "bankroll": bankroll}


def score_strategy_sizing(matrix_rows: Sequence[Mapping[str, Any]], strategy: str, sizing: str) -> dict[str, Any]:
    subset = [row for row in matrix_rows if row.get("strategy") == strategy and row.get("sizing") == sizing]
    if not subset:
        return {}
    min_return = min(float(row.get("return_pct", 0.0)) for row in subset)
    max_dd = max(float(row.get("max_drawdown_pct", 0.0)) for row in subset)
    min_trades = min(int(row.get("trades", 0)) for row in subset)
    min_final = min(float(row.get("final_balance", 0.0)) for row in subset)
    avg_return = mean(float(row.get("return_pct", 0.0)) for row in subset)
    avg_profit_factor = mean(float(row.get("profit_factor", 0.0)) for row in subset)
    score = float(min_return - 0.75 * max_dd + 2.5 * math.log1p(max(0, min_trades)) + 3.0 * min(avg_profit_factor, 5.0))
    return {
        "strategy": strategy,
        "sizing": sizing,
        "min_return_pct": float(min_return),
        "avg_return_pct": float(avg_return),
        "max_drawdown_pct": float(max_dd),
        "min_trades": int(min_trades),
        "min_final_balance": float(min_final),
        "avg_profit_factor": float(avg_profit_factor),
        "score": score,
    }


def choose_candidates(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    bankroll_rows = list(matrix.get("bankroll") or [])
    pairs = sorted({(str(row.get("strategy")), str(row.get("sizing"))) for row in bankroll_rows})
    scored = [score_strategy_sizing(bankroll_rows, strategy, sizing) for strategy, sizing in pairs]
    scored = [row for row in scored if row]
    scored.sort(key=lambda row: float(row.get("score", -1e9)), reverse=True)
    return scored


def legacy_replay_assumptions() -> dict[str, Any]:
    return {
        "replay_mode": "legacy_bid_maker_independent_fill_research",
        "runtime_parity": False,
        "warning": (
            "This replay is a legacy research artifact. It is not runtime-equivalent "
            "forward evidence and should not be used as paper/live promotion proof."
        ),
        "selection_window_role": "same_window_candidate_ranking",
        "candidate_selection_bias": "strategy and sizing are ranked on the same rows shown in the report",
        "entry_price_semantics": "selected-side best bid by default",
        "order_type_semantics": "maker-oriented legacy replay",
        "fill_model": "independent fill unless adverse-fill stress override is applied",
        "fee_semantics": "config fee path by default; recommended candidate historically assumed maker zero-fee execution",
        "liquidity_cap_model": "none in legacy replay path",
        "row_level_taker_fee_parity": False,
        "top3_depth_cap_parity": False,
        "strict_taker_parity": False,
    }


def build_replay_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    empirical_probs: Sequence[float | None] | None = None,
    strategies: Sequence[StrategySpec] = DEFAULT_STRATEGIES,
    sizings: Sequence[SizingSpec] = DEFAULT_SIZINGS,
    bankrolls: Sequence[float] = DEFAULT_BANKROLLS,
    recommended_strategy: str = "proxy_confirmed_pup_edge003",
    recommended_sizing: str = "flat_8pct_cap150",
) -> dict[str, Any]:
    matrix = evaluate_matrix(rows, strategies=strategies, sizings=sizings, bankrolls=bankrolls, empirical_probs=empirical_probs)
    ranking = choose_candidates(matrix)
    strategy_specs = {item.name: asdict(item) for item in strategies}
    sizing_specs = {item.name: asdict(item) for item in sizings}
    recommended_spec = next((item for item in strategies if item.name == recommended_strategy), strategies[0])
    diagnostics = diagnostic_buckets(rows, recommended_spec, empirical_probs=empirical_probs)
    segments = segments_12h(rows, recommended_spec, empirical_probs=empirical_probs)
    stress = []
    sizing_spec = next((item for item in sizings if item.name == recommended_sizing), sizings[0])
    for gap in (0.10, 0.15, 0.20):
        for initial in bankrolls:
            stress.append(
                simulate_bankroll(
                    rows,
                    recommended_spec,
                    sizing_spec,
                    initial_balance=float(initial),
                    empirical_probs=empirical_probs,
                    win_fill_factor=1.0 - gap,
                    loss_fill_factor=1.0,
                )
            )
    recommended_bankroll = [
        row
        for row in matrix["bankroll"]
        if row.get("strategy") == recommended_strategy and row.get("sizing") == recommended_sizing
    ]
    fixed_by_name = {row["strategy"]: row for row in matrix["fixed_cash_100"]}
    return {
        "assumptions": legacy_replay_assumptions(),
        "rows": int(len(rows)),
        "first_created_at": str(rows[0].get("created_at") if rows else None),
        "last_created_at": str(rows[-1].get("created_at") if rows else None),
        "actual_side_counts": dict(Counter(str(row.get("actual_side")) for row in rows)),
        "strategy_specs": strategy_specs,
        "sizing_specs": sizing_specs,
        "fixed_cash_100": matrix["fixed_cash_100"],
        "bankroll": matrix["bankroll"],
        "ranking": ranking,
        "recommended": {
            "strategy": recommended_strategy,
            "sizing": recommended_sizing,
            "fixed_cash_100": fixed_by_name.get(recommended_strategy),
            "bankroll": recommended_bankroll,
            "adverse_fill_stress": stress,
            "diagnostics": diagnostics,
            "segments_12h": segments,
        },
    }
