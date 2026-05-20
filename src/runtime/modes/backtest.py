"""Backtest entrypoints: simple Brier/edge replay and walk-forward."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.backtest.walkforward import WalkForwardConfig, run_walkforward_backtest
from src.polymarket.fees import FeeModelConfig
from src.runtime.persistence import read_jsonl, write_json
from src.runtime.utilities import (
    compute_trade_pnl_per_share,
    resolve_output_path,
    to_float,
)
from src.strategy.calibration import clip_prob
from src.utils.logging import build_logger


def run_backtest(cfg: dict[str, Any]) -> dict[str, Any]:
    logger = build_logger(f"{cfg['app']['name']}_backtest", cfg["paths"]["logs_dir"])

    mon_cfg = cfg.get("monitor", {})
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outcomes_path = resolve_output_path(outputs_dir, str(mon_cfg.get("outcomes_file", "outcomes_5m.jsonl")))
    rows = read_jsonl(outcomes_path)
    if len(rows) < 10:
        return {"error": "not_enough_outcomes", "rows": len(rows), "path": str(outcomes_path)}

    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        side = str(row.get("actual_side", "")).strip().lower()
        if side not in {"up", "down"}:
            continue
        try:
            p_up = clip_prob(float(row.get("p_up")))
        except Exception:
            continue
        eval_rows.append({"p_up": p_up, "actual_side": side, **row})

    n = len(eval_rows)
    if n < 10:
        return {"error": "not_enough_valid_rows", "rows": n}

    y = [1.0 if r["actual_side"] == "up" else 0.0 for r in eval_rows]
    p = [float(r["p_up"]) for r in eval_rows]
    preds = [1.0 if x >= 0.5 else 0.0 for x in p]

    accuracy = sum(1.0 for yy, pp in zip(y, preds, strict=True) if yy == pp) / n
    up_rate = sum(y) / n
    brier = sum((pp - yy) ** 2 for pp, yy in zip(p, y, strict=True)) / n
    brier_p05 = sum((0.5 - yy) ** 2 for yy in y) / n
    brier_up_rate = sum((up_rate - yy) ** 2 for yy in y) / n

    fee_cfg = FeeModelConfig(**cfg["fees"])
    min_edge_to_trade = float(cfg["execution"]["min_edge_to_trade"])
    taker_slippage_bps = float(cfg["execution"]["slippage_bps_taker"])

    priced = 0
    traded = 0
    pnl_rows: list[float] = []
    edge_rows: list[float] = []
    for row in eval_rows:
        up_ask = to_float(row.get("up_best_ask"), -1.0)
        down_ask = to_float(row.get("down_best_ask"), -1.0)
        if not (0.0 < up_ask < 1.0 and 0.0 < down_ask < 1.0):
            continue

        priced += 1
        predicted_side = str(row.get("predicted_side", ""))
        taker_fee_override = to_float(
            row.get("up_taker_fee_bps") if predicted_side == "up" else row.get("down_taker_fee_bps"),
            float(fee_cfg.taker_fee_bps),
        )
        pnl_info = compute_trade_pnl_per_share(
            predicted_side=predicted_side,
            actual_side=str(row["actual_side"]),
            p_up=float(row["p_up"]),
            up_ask=up_ask,
            down_ask=down_ask,
            fee_cfg=fee_cfg,
            taker_slippage_bps=taker_slippage_bps,
            taker_fee_bps_override=taker_fee_override,
        )
        edge = float(pnl_info["edge"])
        if edge < min_edge_to_trade:
            continue

        traded += 1
        edge_rows.append(edge)
        pnl_rows.append(float(pnl_info["net_pnl"]))

    avg_edge = (sum(edge_rows) / len(edge_rows)) if edge_rows else 0.0
    net_pnl = sum(pnl_rows) if pnl_rows else 0.0
    mean_pnl = (net_pnl / len(pnl_rows)) if pnl_rows else 0.0
    win_rate = (sum(1 for x in pnl_rows if x > 0) / len(pnl_rows)) if pnl_rows else 0.0

    payload = {
        "rows_total": len(rows),
        "rows_evaluated": n,
        "classification": {
            "accuracy": accuracy,
            "up_rate": up_rate,
            "pred_up_rate": sum(preds) / n,
            "mean_p_up": sum(p) / n,
            "brier": brier,
            "brier_baseline_p05": brier_p05,
            "brier_baseline_up_rate": brier_up_rate,
        },
        "edge_sim_taker_1share": {
            "rows_with_entry_prices": priced,
            "trades_taken": traded,
            "avg_model_edge": avg_edge,
            "net_pnl": net_pnl,
            "mean_pnl": mean_pnl,
            "win_rate": win_rate,
        },
    }
    logger.info("backtest_complete", extra={"extra": payload})
    out_path = Path(cfg["paths"]["outputs_dir"]) / "backtest_summary.json"
    write_json(out_path, payload)
    return payload


def run_backtest_walkforward(cfg: dict[str, Any]) -> dict[str, Any]:
    logger = build_logger(f"{cfg['app']['name']}_backtest_wf", cfg["paths"]["logs_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    backtest_cfg = cfg.get("backtest", {})
    wf_cfg_raw = backtest_cfg.get("walkforward", {}) if isinstance(backtest_cfg, dict) else {}
    source_file_raw = wf_cfg_raw.get("source_outcomes_file")
    if source_file_raw:
        outcomes_path = resolve_output_path(outputs_dir, str(source_file_raw))
        if not outcomes_path.exists():
            return {"error": "source_outcomes_not_found", "path": str(outcomes_path)}
    else:
        outcomes_path = resolve_output_path(outputs_dir, "paper_outcomes_5m.jsonl")
        if not outcomes_path.exists():
            outcomes_path = resolve_output_path(
                outputs_dir, str(cfg.get("monitor", {}).get("outcomes_file", "outcomes_5m.jsonl"))
            )

    rows = read_jsonl(outcomes_path)
    if len(rows) < 20:
        return {"error": "not_enough_outcomes", "rows": len(rows), "path": str(outcomes_path)}

    wf_cfg = WalkForwardConfig(
        lookback=int(wf_cfg_raw.get("lookback", 240)),
        min_train_samples=int(wf_cfg_raw.get("min_train_samples", 40)),
        min_edge_to_trade=float(wf_cfg_raw.get("min_edge_to_trade", cfg["execution"]["min_edge_to_trade"])),
        min_edge_for_taker=float(wf_cfg_raw.get("min_edge_for_taker", cfg["execution"]["min_edge_for_taker"])),
        max_exposure_usd=float(wf_cfg_raw.get("max_exposure_usd", cfg["risk"]["max_exposure_per_window_usd"])),
        maker_preference=bool(wf_cfg_raw.get("maker_preference", cfg["execution"].get("maker_preference", True))),
        maker_fill_probability=float(wf_cfg_raw.get("maker_fill_probability", cfg["execution"].get("maker_fill_probability", 0.65))),
        maker_ev_advantage_required=float(wf_cfg_raw.get("maker_ev_advantage_required", cfg["execution"].get("maker_ev_advantage_required", 0.0005))),
        maker_slippage_bps=float(wf_cfg_raw.get("maker_slippage_bps", cfg["execution"].get("slippage_bps_maker", 3.0))),
        maker_fill_floor=float(wf_cfg_raw.get("maker_fill_floor", 0.05)),
        maker_fill_cap=float(wf_cfg_raw.get("maker_fill_cap", 0.9)),
        maker_fill_base=float(wf_cfg_raw.get("maker_fill_base", 0.78)),
        maker_fill_spread_penalty=float(wf_cfg_raw.get("maker_fill_spread_penalty", 7.0)),
        maker_fill_late_penalty_90=float(wf_cfg_raw.get("maker_fill_late_penalty_90", 0.15)),
        maker_fill_late_penalty_45=float(wf_cfg_raw.get("maker_fill_late_penalty_45", 0.10)),
        require_book_quality=bool(wf_cfg_raw.get("require_book_quality", False)),
        max_outcome_spread=float(wf_cfg_raw.get("max_outcome_spread", 1.0)),
        allowed_order_types=tuple(
            str(x).strip().lower()
            for x in wf_cfg_raw.get("allowed_order_types", cfg["execution"].get("allowed_order_types", ["maker", "taker"]))
            if str(x).strip().lower() in {"maker", "taker"}
        ) or ("maker", "taker"),
    )
    fee_cfg = FeeModelConfig(**cfg["fees"])
    payload = run_walkforward_backtest(
        rows=rows,
        wf_cfg=wf_cfg,
        fee_cfg=fee_cfg,
        taker_slippage_bps=float(cfg["execution"]["slippage_bps_taker"]),
    )
    payload["source_path"] = str(outcomes_path)
    logger.info("backtest_walkforward_complete", extra={"extra": payload})
    out_path = Path(cfg["paths"]["outputs_dir"]) / "backtest_walkforward_summary.json"
    write_json(out_path, payload)
    return payload
