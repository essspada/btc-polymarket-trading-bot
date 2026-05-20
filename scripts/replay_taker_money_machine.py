#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy.actionability_calibration import evaluate_actionability_calibration

DEFAULT_DATASETS = [
    (
        "sample_apr29",
        "bundled sample (~500 windows starting 2026-04-29)",
        REPO_ROOT / "data/sample/outcomes_5m.jsonl",
    ),
]
DEFAULT_BANKROLLS = "100,200,350,500,750,1000"
DEFAULT_MIN_EDGE = 0.01
DEFAULT_PROXY_MIN_EDGE = 0.0
DEFAULT_MIN_NET_EDGE = 0.01
DEFAULT_MAX_SPREAD = 0.02
DEFAULT_MIN_PRICE = 0.01
DEFAULT_MAX_PRICE = 0.70
DEFAULT_FEE_SOURCE = "row"


@dataclass(frozen=True)
class Trade:
    side: str
    actual_side: str
    bid_price: float
    ask_price: float
    gate_price: float
    spread: float
    top3_ask_size: float | None
    ts: str
    fee_rate: float | None = None
    raw_p_side: float | None = None
    calibrated_p_side: float | None = None
    calibrated_net_edge: float | None = None
    calibration_reason: str | None = None


@dataclass(frozen=True)
class ReplayConfig:
    min_stage_delay: int = 240
    min_edge: float = DEFAULT_MIN_EDGE
    proxy_min_edge: float = DEFAULT_PROXY_MIN_EDGE
    min_net_edge: float = DEFAULT_MIN_NET_EDGE
    max_spread: float = DEFAULT_MAX_SPREAD
    min_price: float = DEFAULT_MIN_PRICE
    max_price: float = DEFAULT_MAX_PRICE
    gate_price: str = "ask"
    fee_rate: float = 0.072
    fee_source: str = DEFAULT_FEE_SOURCE
    slippage_bps: float = 12.0
    risk_fraction: float = 0.08
    max_trade_usd: float = 150.0
    min_trade_usd: float = 5.0
    liquidity_haircut: float = 0.95
    min_top3_ask_size: float | None = None
    actionability_calibration: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReplayResult:
    initial_bankroll: float
    final_balance: float
    return_pct: float
    max_drawdown_pct: float
    executed_trades: int
    correct_trades: int
    win_rate_pct: float
    liquidity_capped_trades: int
    skipped_min_trade: int


@dataclass(frozen=True)
class DatasetResult:
    name: str
    window: str
    path: Path
    selected_trades: int
    selected_win_rate_pct: float
    rows: list[ReplayResult]


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


def _clamp_prob(value: Any) -> float | None:
    out = _to_float(value)
    if out is None:
        return None
    return float(max(1e-6, min(1.0 - 1e-6, out)))


def extract_prob(row: dict[str, Any], key: str) -> float | None:
    direct = _clamp_prob(row.get(key))
    if direct is not None:
        return direct
    for suffix in ("_p_up", ""):
        value = _clamp_prob(row.get(f"{key}{suffix}"))
        if value is not None:
            return value
    model = row.get("candidate_models", {}).get(key) if isinstance(row.get("candidate_models"), dict) else None
    if isinstance(model, dict):
        for subkey in ("p_up", "p_up_calibrated", "p_up_raw"):
            value = _clamp_prob(model.get(subkey))
            if value is not None:
                return value
    return None


def _stage_delay(row: dict[str, Any]) -> int | None:
    for key in ("timing_policy_stage_delay", "spot_timing_policy_stage_delay"):
        delay = _to_int(row.get(key))
        if delay is not None:
            return delay
    models = row.get("candidate_models")
    if isinstance(models, dict):
        for model in models.values():
            if isinstance(model, dict):
                delay = _to_int(model.get("timing_policy_stage_delay"))
                if delay is not None:
                    return delay
    return None


def _side_prices(row: dict[str, Any], side: str, gate_price: str) -> tuple[float, float, float, float, float | None] | None:
    bid = _to_float(row.get(f"{side}_best_bid"))
    ask = _to_float(row.get(f"{side}_best_ask"))
    if bid is None or ask is None or not (0.0 < bid < 1.0) or not (0.0 < ask < 1.0):
        return None
    spread = _to_float(row.get(f"{side}_spread"), ask - bid)
    if spread is None:
        spread = ask - bid
    gate = bid if gate_price == "bid" else ask
    top3_ask = _to_float(row.get(f"{side}_top3_ask_size"))
    return float(bid), float(ask), float(gate), float(max(0.0, spread)), top3_ask


def _row_fee_rate(row: dict[str, Any], side: str, cfg: ReplayConfig) -> float:
    if str(cfg.fee_source).strip().lower() == "row":
        bps = _to_float(row.get(f"{side}_taker_fee_bps"))
        if bps is not None:
            return max(0.0, float(bps) / 10_000.0)
    return max(0.0, float(cfg.fee_rate))


def replay_config_from_runtime(cfg: Mapping[str, Any]) -> ReplayConfig:
    defaults = ReplayConfig(gate_price="ask")
    exec_cfg = cfg.get("execution", {}) if isinstance(cfg.get("execution"), Mapping) else {}
    gate_cfg = exec_cfg.get("confirmation_gate", {}) if isinstance(exec_cfg.get("confirmation_gate"), Mapping) else {}
    actionability_cfg = (
        exec_cfg.get("actionability_calibration")
        if isinstance(exec_cfg.get("actionability_calibration"), Mapping)
        else {}
    )
    fees_cfg = cfg.get("fees", {}) if isinstance(cfg.get("fees"), Mapping) else {}
    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk"), Mapping) else {}

    return ReplayConfig(
        gate_price="ask",
        min_edge=_coalesce(_to_float(exec_cfg.get("min_edge_to_trade")), defaults.min_edge),
        proxy_min_edge=_coalesce(_to_float(gate_cfg.get("min_edge")), defaults.proxy_min_edge),
        min_net_edge=_coalesce(_to_float(exec_cfg.get("min_edge_to_trade")), defaults.min_net_edge),
        max_spread=_coalesce(_to_float(gate_cfg.get("max_spread")), defaults.max_spread),
        min_price=_coalesce(_to_float(gate_cfg.get("min_price")), defaults.min_price),
        max_price=_coalesce(_to_float(exec_cfg.get("max_entry_price")), defaults.max_price),
        fee_rate=_coalesce(_to_float(fees_cfg.get("taker_fee_rate")), defaults.fee_rate),
        fee_source=str(fees_cfg.get("taker_fee_source") or defaults.fee_source),
        slippage_bps=_coalesce(_to_float(exec_cfg.get("slippage_bps_taker")), defaults.slippage_bps),
        risk_fraction=_coalesce(_to_float(risk_cfg.get("max_balance_fraction_per_trade")), defaults.risk_fraction),
        max_trade_usd=_coalesce(_to_float(risk_cfg.get("max_exposure_per_window_usd")), defaults.max_trade_usd),
        min_trade_usd=_coalesce(_to_float(risk_cfg.get("min_trade_usd")), defaults.min_trade_usd),
        liquidity_haircut=_coalesce(_to_float(exec_cfg.get("liquidity_haircut")), defaults.liquidity_haircut),
        min_stage_delay=_coalesce(_to_int(exec_cfg.get("min_stage_delay_seconds")), defaults.min_stage_delay),
        min_top3_ask_size=_to_float(gate_cfg.get("min_top3_ask_size")),
        actionability_calibration=(dict(actionability_cfg) if bool(actionability_cfg.get("enabled", False)) else None),
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            actual = str(row.get("actual_side", "")).strip().lower()
            if actual not in {"up", "down"}:
                continue
            row["actual_side"] = actual
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("created_at") or row.get("ts_utc") or ""))
    return rows


def select_trades(rows: Iterable[dict[str, Any]], cfg: ReplayConfig) -> list[Trade]:
    trades: list[Trade] = []
    for row in rows:
        delay = _stage_delay(row)
        if delay is None or delay < cfg.min_stage_delay:
            continue

        p_up = extract_prob(row, "spot_logistic_online")
        proxy_p_up = extract_prob(row, "proxy_logistic_market_blend")
        if p_up is None or proxy_p_up is None:
            continue

        side = "up" if p_up >= 0.5 else "down"
        proxy_side = "up" if proxy_p_up >= 0.5 else "down"
        if proxy_side != side:
            continue

        prices = _side_prices(row, side, cfg.gate_price)
        if prices is None:
            continue
        bid, ask, gate, spread, top3_ask = prices
        if not (cfg.min_price <= gate <= cfg.max_price):
            continue
        if spread > cfg.max_spread:
            continue
        if cfg.min_top3_ask_size is not None and (
            top3_ask is None or float(top3_ask) < float(cfg.min_top3_ask_size)
        ):
            continue

        p_side = p_up if side == "up" else 1.0 - p_up
        proxy_p_side = proxy_p_up if side == "up" else 1.0 - proxy_p_up
        if p_side - gate < cfg.min_edge:
            continue
        if proxy_p_side - gate < cfg.proxy_min_edge:
            continue
        fee_rate = _row_fee_rate(row, side, cfg)
        cash_per_share, _, _ = _cash_per_share(ask, cfg, fee_rate=fee_rate)
        net_edge = float(p_side - cash_per_share)
        calibrated_p_side = p_side
        calibration_reason: str | None = None
        actionability_cfg = cfg.actionability_calibration
        if isinstance(actionability_cfg, Mapping) and bool(actionability_cfg.get("enabled", False)):
            calibration = evaluate_actionability_calibration(
                raw_p_side=float(p_side),
                cash_per_share=float(cash_per_share),
                breakeven_probability=float(cash_per_share),
                size_shares=1.0,
                fill_probability=1.0,
                cash_required=float(cash_per_share),
                context={
                    "selected_side": side,
                    "selected_price": ask,
                    "selected_spread": spread,
                    "selected_top3_ask_size": top3_ask,
                    "selected_top3_bid_size": _to_float(row.get(f"{side}_top3_bid_size")),
                    "confidence": max(p_up, 1.0 - p_up),
                    "model_proxy_gap": float(p_up - proxy_p_up),
                    "decision_best_edge": float(p_side - ask),
                    "decision_expected_roi_cash": float((p_side - cash_per_share) / cash_per_share) if cash_per_share > 0.0 else None,
                    "decision_breakeven_margin": float(p_side - cash_per_share),
                    "spot_recent_vol_5m_bps": _to_float(row.get("spot_recent_vol_5m_bps")),
                },
                calibration_cfg=actionability_cfg,
            )
            if calibration is not None:
                calibrated_p_side = float(calibration.calibrated_p_side)
                calibration_reason = calibration.reason
                if calibration.rejected:
                    continue
                if calibration.calibrated_net_edge is not None:
                    net_edge = float(calibration.calibrated_net_edge)
        if net_edge < float(cfg.min_net_edge):
            continue

        trades.append(
            Trade(
                side=side,
                actual_side=str(row["actual_side"]),
                bid_price=bid,
                ask_price=ask,
                gate_price=gate,
                spread=spread,
                top3_ask_size=top3_ask,
                ts=str(row.get("created_at") or row.get("ts_utc") or ""),
                fee_rate=fee_rate,
                raw_p_side=float(p_side),
                calibrated_p_side=calibrated_p_side,
                calibrated_net_edge=float(net_edge),
                calibration_reason=calibration_reason,
            )
        )
    return trades


def _cash_per_share(ask_price: float, cfg: ReplayConfig, *, fee_rate: float | None = None) -> tuple[float, float, float]:
    effective_fee_rate = float(cfg.fee_rate if fee_rate is None else fee_rate)
    fee_per_share = effective_fee_rate * float(ask_price) * (1.0 - float(ask_price))
    slippage_per_share = float(ask_price) * (float(cfg.slippage_bps) / 10_000.0)
    return float(ask_price) + fee_per_share + slippage_per_share, fee_per_share, slippage_per_share


def simulate_bankroll(trades: Iterable[Trade], initial_bankroll: float, cfg: ReplayConfig) -> ReplayResult:
    balance = float(initial_bankroll)
    peak = balance
    max_drawdown = 0.0
    executed = 0
    correct = 0
    liquidity_capped = 0
    skipped_min_trade = 0

    for trade in trades:
        if balance <= 0.0:
            break
        cash_budget = min(balance, balance * float(cfg.risk_fraction), float(cfg.max_trade_usd))
        if cash_budget <= 0.0:
            continue

        cash_per_share, fee_per_share, slippage_per_share = _cash_per_share(
            trade.ask_price,
            cfg,
            fee_rate=trade.fee_rate,
        )
        if cash_per_share <= 0.0:
            continue
        budget_shares = cash_budget / cash_per_share

        top3 = trade.top3_ask_size
        if top3 is None:
            liquidity_shares = float("inf")
        else:
            liquidity_shares = max(0.0, float(top3)) * float(cfg.liquidity_haircut)
        shares = min(budget_shares, liquidity_shares)
        if shares + 1e-12 < budget_shares:
            liquidity_capped += 1

        cash_required = shares * cash_per_share
        if cash_required < float(cfg.min_trade_usd):
            skipped_min_trade += 1
            continue

        fee = shares * fee_per_share
        slippage = shares * slippage_per_share
        payout = 1.0 if trade.side == trade.actual_side else 0.0
        pnl = shares * (payout - trade.ask_price) - fee - slippage
        balance += pnl

        executed += 1
        if trade.side == trade.actual_side:
            correct += 1
        peak = max(peak, balance)
        if peak > 0.0:
            max_drawdown = max(max_drawdown, (peak - balance) / peak)

    return ReplayResult(
        initial_bankroll=float(initial_bankroll),
        final_balance=float(balance),
        return_pct=float((balance / float(initial_bankroll) - 1.0) * 100.0),
        max_drawdown_pct=float(max_drawdown * 100.0),
        executed_trades=executed,
        correct_trades=correct,
        win_rate_pct=float((correct / executed) * 100.0) if executed else 0.0,
        liquidity_capped_trades=liquidity_capped,
        skipped_min_trade=skipped_min_trade,
    )


def run_dataset(name: str, window: str, path: Path, bankrolls: list[float], cfg: ReplayConfig) -> DatasetResult:
    rows = load_rows(path)
    trades = select_trades(rows, cfg)
    wins = sum(1 for trade in trades if trade.side == trade.actual_side)
    selected_wr = (wins / len(trades) * 100.0) if trades else 0.0
    return DatasetResult(
        name=name,
        window=window,
        path=path,
        selected_trades=len(trades),
        selected_win_rate_pct=float(selected_wr),
        rows=[simulate_bankroll(trades, bankroll, cfg) for bankroll in bankrolls],
    )


def _format_money(value: float) -> str:
    return f"${value:.2f}"


def _format_bankroll(value: float) -> str:
    return f"${value:.0f}" if float(value).is_integer() else f"${value:.2f}"


def _format_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _box_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: list[str]) -> str:
        return "│ " + " │ ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " │"

    top = "┌" + "┬".join("─" * (width + 2) for width in widths) + "┐"
    mid = "├" + "┼".join("─" * (width + 2) for width in widths) + "┤"
    bottom = "└" + "┴".join("─" * (width + 2) for width in widths) + "┘"
    return "\n".join([top, fmt_row(headers), mid, *(fmt_row(row) for row in rows), bottom])


def print_dataset_result(result: DatasetResult) -> None:
    print(
        f"{result.name} ({result.window}, {result.selected_trades} сигналов, "
        f"WR {result.selected_win_rate_pct:.1f}%):"
    )
    rows = [
        [
            _format_bankroll(row.initial_bankroll),
            _format_money(row.final_balance),
            _format_pct(row.return_pct),
            f"{row.max_drawdown_pct:.1f}%",
            str(row.executed_trades),
            str(row.liquidity_capped_trades),
        ]
        for row in result.rows
    ]
    print(_box_table(["Банкролл", "Итоговый баланс", "Чистый %", "Макс. просадка", "Сделок", "Liq cap"], rows))
    print()


def _parse_bankrolls(raw: str) -> list[float]:
    bankrolls: list[float] = []
    for part in raw.split(","):
        value = _to_float(part.strip())
        if value is not None and value > 0.0:
            bankrolls.append(value)
    if not bankrolls:
        raise ValueError("at least one positive bankroll is required")
    return bankrolls


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay the taker Money Machine strategy on recorded 5m BTC outcomes.")
    parser.add_argument("--bankrolls", default=DEFAULT_BANKROLLS)
    parser.add_argument("--gate-price", choices=("ask", "bid"), default="ask")
    parser.add_argument("--fee-rate", type=float, default=0.072)
    parser.add_argument("--slippage-bps", type=float, default=12.0)
    parser.add_argument("--risk-fraction", type=float, default=0.08)
    parser.add_argument("--max-trade-usd", type=float, default=150.0)
    parser.add_argument("--min-trade-usd", type=float, default=5.0)
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    parser.add_argument("--proxy-min-edge", type=float, default=DEFAULT_PROXY_MIN_EDGE)
    parser.add_argument("--min-net-edge", type=float, default=DEFAULT_MIN_NET_EDGE)
    parser.add_argument("--max-spread", type=float, default=DEFAULT_MAX_SPREAD)
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    parser.add_argument("--max-price", type=float, default=DEFAULT_MAX_PRICE)
    parser.add_argument("--fee-source", choices=("row", "config"), default=DEFAULT_FEE_SOURCE)
    parser.add_argument("--min-stage-delay", type=int, default=240)
    parser.add_argument("--liquidity-haircut", type=float, default=0.95)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    bankrolls = _parse_bankrolls(args.bankrolls)
    cfg = ReplayConfig(
        min_stage_delay=int(args.min_stage_delay),
        min_edge=float(args.min_edge),
        proxy_min_edge=float(args.proxy_min_edge),
        min_net_edge=float(args.min_net_edge),
        max_spread=float(args.max_spread),
        min_price=float(args.min_price),
        max_price=float(args.max_price),
        gate_price=str(args.gate_price),
        fee_rate=float(args.fee_rate),
        fee_source=str(args.fee_source),
        slippage_bps=float(args.slippage_bps),
        risk_fraction=float(args.risk_fraction),
        max_trade_usd=float(args.max_trade_usd),
        min_trade_usd=float(args.min_trade_usd),
        liquidity_haircut=float(args.liquidity_haircut),
    )
    print(
        "Replay config: "
        f"gate_price={cfg.gate_price}, fee_rate={cfg.fee_rate}, slippage_bps={cfg.slippage_bps}, "
        f"risk={cfg.risk_fraction:.1%}, max_trade=${cfg.max_trade_usd:.0f}, "
        f"min_edge={cfg.min_edge:.3f}, proxy_min_edge={cfg.proxy_min_edge:.3f}, "
        f"min_net_edge={cfg.min_net_edge:.3f}, fee_source={cfg.fee_source}, "
        f"max_spread={cfg.max_spread:.3f}, price=[{cfg.min_price:.2f},{cfg.max_price:.2f}], "
        f"liquidity_haircut={cfg.liquidity_haircut:.0%}"
    )
    print()
    for name, window, path in DEFAULT_DATASETS:
        if not path.exists():
            print(f"{name}: file not found: {path}")
            continue
        print_dataset_result(run_dataset(name, window, path, bankrolls, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
