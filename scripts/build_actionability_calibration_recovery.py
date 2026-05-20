#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_taker_money_machine import (
    ReplayConfig,
    _cash_per_share,
    load_rows,
    select_trades,
    simulate_bankroll,
)
from src.strategy.actionability_calibration import wilson_lower_bound

DATASETS: dict[str, Path] = {
    "historical": ROOT / "outputs/monitor_only_ag_shadow/recovery_ag_monitor_20260429_020344/outcomes_5m.jsonl",
    "may6": ROOT / "outputs/monitor_only_shadow_rollout/shadow_monitor_money_machine_20260506_051848/outcomes_5m.jsonl",
    "may8": ROOT / "outputs/monitor_only_shadow_rollout/shadow_monitor_money_machine_20260508_202451/outcomes_5m.jsonl",
    "liq150_stopped": ROOT / "outputs/monitor_only_shadow_rollout/shadow_monitor_money_machine_liq150_20260512_080940/outcomes_5m.jsonl",
}
MAY8_RUNTIME_CONFIG = ROOT / "outputs/monitor_only_shadow_rollout/shadow_monitor_money_machine_20260508_202451/monitor_only_ag_config.yaml"

FEATURE_BOUNDS: dict[str, list[float]] = {
    "confidence": [0.50, 0.55, 0.60, 0.65, 0.70, 1.01],
    "selected_price": [0.0, 0.40, 0.50, 0.60, 0.70, 1.01],
    "selected_spread": [0.0, 0.01, 0.02, 0.03, 1.01],
    "selected_top3_ask_size": [0.0, 100.0, 300.0, 700.0, 1500.0, 1.0e12],
    "model_proxy_gap": [0.0, 0.01, 0.03, 0.06, 1.0],
    "decision_breakeven_margin": [-1.0, 0.0, 0.01, 0.03, 0.06, 1.0],
    "spot_recent_vol_5m_bps": [0.0, 5.0, 10.0, 20.0, 40.0, 1.0e12],
}


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_may8_cfg() -> ReplayConfig:
    cfg = yaml.safe_load(MAY8_RUNTIME_CONFIG.read_text(encoding="utf-8")) or {}
    exec_cfg = cfg.get("execution", {}) if isinstance(cfg.get("execution"), Mapping) else {}
    gate_cfg = exec_cfg.get("confirmation_gate", {}) if isinstance(exec_cfg.get("confirmation_gate"), Mapping) else {}
    fees_cfg = cfg.get("fees", {}) if isinstance(cfg.get("fees"), Mapping) else {}
    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk"), Mapping) else {}
    return ReplayConfig(
        gate_price="ask",
        min_edge=float(_float(exec_cfg.get("min_edge_to_trade"), 0.01)),
        proxy_min_edge=float(_float(gate_cfg.get("min_edge"), 0.0)),
        min_net_edge=float(_float(exec_cfg.get("min_edge_to_trade"), 0.01)),
        max_spread=float(_float(gate_cfg.get("max_spread"), 0.02)),
        min_price=float(_float(gate_cfg.get("min_price"), 0.01)),
        max_price=float(_float(exec_cfg.get("max_entry_price"), 0.70)),
        fee_rate=float(_float(fees_cfg.get("taker_fee_rate"), 0.072)),
        fee_source=str(fees_cfg.get("taker_fee_source") or "row"),
        slippage_bps=float(_float(exec_cfg.get("slippage_bps_taker"), 12.0)),
        risk_fraction=float(_float(risk_cfg.get("max_balance_fraction_per_trade"), 0.08)),
        max_trade_usd=float(_float(risk_cfg.get("max_exposure_per_window_usd"), 150.0)),
        min_trade_usd=float(_float(risk_cfg.get("min_trade_usd"), 5.0)),
        liquidity_haircut=float(_float(exec_cfg.get("liquidity_haircut"), 0.95)),
        min_stage_delay=int(_float(exec_cfg.get("min_stage_delay_seconds"), 240.0) or 240),
        min_top3_ask_size=_float(gate_cfg.get("min_top3_ask_size")),
    )


def _extract_prob(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        out = float(value)
        if 0.0 <= out <= 1.0:
            return out
    model = row.get("candidate_models")
    if isinstance(model, Mapping):
        item = model.get(key)
        if isinstance(item, Mapping):
            for subkey in ("p_up", "p_up_calibrated", "p_up_raw"):
                out = _float(item.get(subkey))
                if out is not None and 0.0 <= out <= 1.0:
                    return out
    return None


def _build_samples(rows: list[dict[str, Any]], replay_cfg: ReplayConfig) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        selected = select_trades([row], replay_cfg)
        if not selected:
            continue
        trade = selected[0]
        side = trade.side
        actual = str(row.get("actual_side") or "").strip().lower()
        if actual not in {"up", "down"}:
            continue
        p_up = _extract_prob(row, "spot_logistic_online")
        proxy_p_up = _extract_prob(row, "proxy_logistic_market_blend")
        if p_up is None or proxy_p_up is None:
            continue
        ask = _float(row.get(f"{side}_best_ask"))
        bid = _float(row.get(f"{side}_best_bid"))
        spread = _float(row.get(f"{side}_spread"), (ask - bid) if ask is not None and bid is not None else None)
        if ask is None or spread is None:
            continue
        p_side = p_up if side == "up" else 1.0 - p_up
        cash_per_share, _, _ = _cash_per_share(ask, replay_cfg, fee_rate=trade.fee_rate)
        confidence = p_up if side == "up" else (1.0 - p_up)
        samples.append(
            {
                "win": 1 if side == actual else 0,
                "confidence": float(confidence),
                "selected_price": float(ask),
                "selected_spread": float(spread),
                "selected_top3_ask_size": _float(row.get(f"{side}_top3_ask_size")),
                "model_proxy_gap": abs(float(p_up - proxy_p_up)),
                "decision_breakeven_margin": float(p_side - cash_per_share),
                "spot_recent_vol_5m_bps": _float(row.get("spot_recent_vol_5m_bps")),
            }
        )
    return samples


def _bucketize(samples: list[dict[str, Any]], feature: str, bounds: list[float], *, z_score: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        chunk = [item for item in samples if item.get(feature) is not None and float(lo) <= float(item[feature]) < float(hi)]
        n = len(chunk)
        wins = sum(int(item["win"]) for item in chunk)
        wr = float(wins / n) if n else None
        lb = wilson_lower_bound(wins=wins, n=n, z=z_score) if n else None
        out.append({
            "lower": float(lo),
            "upper": float(hi),
            "n": int(n),
            "wins": int(wins),
            "wr": wr,
            "lb_wr": lb,
        })
    return out


def build_artifact(
    train_samples: list[dict[str, Any]],
    *,
    min_bucket_samples: int,
    conservative_alpha: float,
    z_score: float,
) -> dict[str, Any]:
    wins = sum(int(item["win"]) for item in train_samples)
    n = len(train_samples)
    global_wr = float(wins / n) if n else None
    global_lb = wilson_lower_bound(wins=wins, n=n, z=z_score) if n else None
    features = {
        feature: {
            "buckets": _bucketize(train_samples, feature, bounds, z_score=z_score),
        }
        for feature, bounds in FEATURE_BOUNDS.items()
    }
    return {
        "schema": "actionability_calibration_v1",
        "objective": "Conservative selected-trade probability lower bound for fee-aware taker economics.",
        "min_bucket_samples": int(min_bucket_samples),
        "conservative_alpha": float(conservative_alpha),
        "global": {
            "n": int(n),
            "wins": int(wins),
            "wr": global_wr,
            "lb_wr": global_lb,
            "z_score": float(z_score),
        },
        "features": features,
        "runtime_fields_only": [
            "confidence",
            "selected_price",
            "selected_spread",
            "selected_top3_ask_size",
            "model_proxy_gap",
            "decision_breakeven_margin",
            "spot_recent_vol_5m_bps",
        ],
    }


def _unit_pnls(trades: list[Any], cfg: ReplayConfig) -> list[float]:
    out: list[float] = []
    for trade in trades:
        cash_per_share, _, _ = _cash_per_share(trade.ask_price, cfg, fee_rate=trade.fee_rate)
        payout = 1.0 if trade.side == trade.actual_side else 0.0
        out.append(float(payout - cash_per_share))
    return out


def _dataset_metrics(rows: list[dict[str, Any]], cfg: ReplayConfig) -> dict[str, Any]:
    trades = select_trades(rows, cfg)
    replay = simulate_bankroll(trades, 1000.0, cfg)
    pnls = _unit_pnls(trades, cfg)
    wins = [value for value in pnls if value > 0.0]
    losses = [value for value in pnls if value < 0.0]
    avg_win = fmean(wins) if wins else None
    avg_loss = fmean(losses) if losses else None
    breakeven_wr = None
    if avg_win is not None and avg_loss is not None and (avg_win + abs(avg_loss)) > 0.0:
        breakeven_wr = abs(avg_loss) / (avg_win + abs(avg_loss))
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    pf = None if gross_losses <= 0.0 else float(gross_wins / gross_losses)
    without_top_winner = float(sum(pnls) - max(pnls)) if pnls else 0.0

    # Monotonicity sanity: top calibrated edge bucket should not underperform low bucket.
    buckets: list[tuple[float, float]] = []
    for trade, pnl in zip(trades, pnls, strict=True):
        edge = trade.calibrated_net_edge if trade.calibrated_net_edge is not None else trade.raw_p_side
        if edge is None:
            continue
        buckets.append((float(edge), float(pnl)))
    buckets.sort(key=lambda item: item[0])
    monotonic_ok = None
    if len(buckets) >= 6:
        split = len(buckets) // 3
        low = fmean(pnl for _, pnl in buckets[:split])
        mid = fmean(pnl for _, pnl in buckets[split : 2 * split])
        high = fmean(pnl for _, pnl in buckets[2 * split :])
        monotonic_ok = bool(high >= low and high >= mid)

    return {
        "selected_trades": int(replay.executed_trades),
        "selected_wr_pct": float(replay.win_rate_pct),
        "final_balance_1000": float(replay.final_balance),
        "return_pct_1000": float(replay.return_pct),
        "max_drawdown_pct_1000": float(replay.max_drawdown_pct),
        "profit_factor": pf,
        "breakeven_wr_pct": (None if breakeven_wr is None else float(breakeven_wr * 100.0)),
        "pnl_without_top_winner": without_top_winner,
        "monotonic_edge_ranking_ok": monotonic_ok,
    }


def _gate(eval_by_dataset: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for name in ("historical", "may6", "may8"):
        m = eval_by_dataset[name]
        if float(m["return_pct_1000"]) <= 0.0:
            reasons.append(f"{name}: return<=0")
        pf = m.get("profit_factor")
        if pf is None or float(pf) < 1.10:
            reasons.append(f"{name}: profit_factor<1.10")
        br = m.get("breakeven_wr_pct")
        if br is None or float(m["selected_wr_pct"]) <= float(br):
            reasons.append(f"{name}: WR<=breakeven")
        if float(m["max_drawdown_pct_1000"]) > 25.0:
            reasons.append(f"{name}: maxDD>25%")
        if float(m["pnl_without_top_winner"]) < 0.0:
            reasons.append(f"{name}: pnl_without_top_winner<0")
        mono = m.get("monotonic_edge_ranking_ok")
        if mono is False:
            reasons.append(f"{name}: monotonic_edge_ranking_inverted")

    sanity = eval_by_dataset["liq150_stopped"]
    if int(sanity["selected_trades"]) < 3:
        reasons.append("liq150_stopped: trades<3 (inconclusive)")
    else:
        if float(sanity["return_pct_1000"]) < 0.0:
            reasons.append("liq150_stopped: return<0")
        pf = sanity.get("profit_factor")
        if pf is None or float(pf) < 1.0:
            reasons.append("liq150_stopped: profit_factor<1.0")

    return {
        "promotable": len(reasons) == 0,
        "reasons": reasons,
    }


def _box_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt(values: list[str]) -> str:
        return "│ " + " │ ".join(values[i].ljust(widths[i]) for i in range(len(headers))) + " │"

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"
    return "\n".join([top, fmt(headers), mid, *(fmt(row) for row in rows), bot])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build conservative actionability calibration artifact and evaluate strict OOS recovery gates.")
    parser.add_argument("--out-artifact", type=Path, default=ROOT / "reports/actionability_calibration_artifact.json")
    parser.add_argument("--out-report", type=Path, default=ROOT / "reports/actionability_calibration_recovery_report.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "reports/actionability_calibration_recovery_report.md")
    parser.add_argument("--min-bucket-samples", type=int, default=20)
    parser.add_argument("--fallback-policy", choices=("reject", "conservative_shrink"), default="conservative_shrink")
    parser.add_argument("--conservative-alpha", type=float, default=0.70)
    parser.add_argument("--min-calibrated-net-edge", type=float, default=0.01)
    parser.add_argument("--min-breakeven-margin", type=float, default=0.0)
    parser.add_argument("--z-score", type=float, default=1.28155, help="Wilson lower-bound z-score (1.28155 ~= 80%% one-sided).")
    args = parser.parse_args()

    baseline_cfg = _load_may8_cfg()
    rows_by_dataset = {name: load_rows(path) for name, path in DATASETS.items()}

    train_samples = _build_samples(rows_by_dataset["historical"], baseline_cfg) + _build_samples(rows_by_dataset["may6"], baseline_cfg)
    artifact = build_artifact(
        train_samples,
        min_bucket_samples=int(args.min_bucket_samples),
        conservative_alpha=float(args.conservative_alpha),
        z_score=float(args.z_score),
    )
    args.out_artifact.parent.mkdir(parents=True, exist_ok=True)
    args.out_artifact.write_text(json.dumps(artifact, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    calibrated_cfg = ReplayConfig(
        **{**asdict(baseline_cfg), "actionability_calibration": {
            "enabled": True,
            "artifact_path": str(args.out_artifact),
            "min_bucket_samples": int(args.min_bucket_samples),
            "fallback_policy": str(args.fallback_policy),
            "conservative_alpha": float(args.conservative_alpha),
            "min_calibrated_net_edge": float(args.min_calibrated_net_edge),
            "min_breakeven_margin": float(args.min_breakeven_margin),
        }}
    )

    baseline_eval = {name: _dataset_metrics(rows, baseline_cfg) for name, rows in rows_by_dataset.items()}
    calibrated_eval = {name: _dataset_metrics(rows, calibrated_cfg) for name, rows in rows_by_dataset.items()}
    gate = _gate(calibrated_eval)

    payload = {
        "schema": "actionability_calibration_recovery_v1",
        "train_split": ["historical", "may6"],
        "main_holdout": "may8",
        "sanity_holdout": "liq150_stopped",
        "artifact_path": str(args.out_artifact),
        "artifact": artifact,
        "baseline_replay_config": asdict(baseline_cfg),
        "calibrated_replay_config": asdict(calibrated_cfg),
        "baseline": baseline_eval,
        "calibrated_candidate": calibrated_eval,
        "acceptance_gate": gate,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    md_rows = []
    for name, metrics in calibrated_eval.items():
        md_rows.append([
            name,
            str(metrics["selected_trades"]),
            f"{float(metrics['selected_wr_pct']):.2f}%",
            f"{float(metrics['return_pct_1000']):+.2f}%",
            f"{float(metrics['max_drawdown_pct_1000']):.2f}%",
            "n/a" if metrics["profit_factor"] is None else f"{float(metrics['profit_factor']):.3f}",
        ])
    lines = [
        "# actionability_calibration_recovery_report",
        "",
        "Calibrated strict taker candidate (offline only, no paper/live authority changes).",
        "",
        _box_table(["Dataset", "Trades", "WR", "Ret1000", "MaxDD", "PF"], md_rows),
        "",
        f"Promotable: {'yes' if gate['promotable'] else 'no'}",
    ]
    for reason in gate["reasons"]:
        lines.append(f"- blocker: {reason}")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(args.out_artifact)
    print(args.out_report)
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
