#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.paper_execution import PaperExecutionEngine, estimate_maker_fill_probability
from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.strategy.signals import decide_trade
from src.strategy.spot_consensus import SpotConsensusConfig, predict_spot_consensus_blend_probability
from src.strategy.spot_logistic import (
    SpotLogisticConfig,
    build_causal_spot_logistic_probability_map,
    fit_rolling_spot_logistic,
    predict_spot_logistic_probability,
    row_key as spot_logistic_row_key,
)
from src.strategy.spot_window_path import (
    SpotWindowPathConfig,
    blend_spot_with_market_probability,
    predict_spot_window_path_probability,
)

CANDIDATES = (
    "proxy_orderbook",
    "spot_window_path",
    "spot_market_blend",
    "spot_logistic_online",
    "spot_logistic_market_blend",
    "spot_consensus_blend",
    "proxy_logistic_market_blend",
    "proxy_logistic_meta_policy",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _clip_prob(value: Any) -> float:
    try:
        prob = float(value)
    except Exception:
        prob = 0.5
    return float(np.clip(prob, 1e-6, 1.0 - 1e-6))


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _valid_price(value: Any) -> bool:
    try:
        value_f = float(value)
    except Exception:
        return False
    return 0.0 < value_f < 1.0


def _build_books_from_row(row: dict[str, Any]) -> MarketBooks | None:
    up_ask = _to_float(row.get("up_best_ask"))
    down_ask = _to_float(row.get("down_best_ask"))
    if not (_valid_price(up_ask) and _valid_price(down_ask)):
        return None
    up_bid = _to_float(row.get("up_best_bid"), 0.0) or 0.0
    down_bid = _to_float(row.get("down_best_bid"), 0.0) or 0.0
    up_mid = _to_float(row.get("up_midpoint"), (up_bid + up_ask) / 2.0 if _valid_price(up_ask) else 0.5)
    down_mid = _to_float(row.get("down_midpoint"), (down_bid + down_ask) / 2.0 if _valid_price(down_ask) else 0.5)
    up_spread = _to_float(row.get("up_spread"), max(0.0, up_ask - up_bid)) or max(0.0, up_ask - up_bid)
    down_spread = _to_float(row.get("down_spread"), max(0.0, down_ask - down_bid)) or max(0.0, down_ask - down_bid)
    return MarketBooks(
        up=OutcomeBook(
            token_id=str(row.get("up_token_id") or "UP"),
            best_bid=float(up_bid),
            best_ask=float(up_ask),
            midpoint=float(up_mid),
            spread=float(up_spread),
            topk_imbalance=float(_to_float(row.get("up_topk_imbalance"), 0.0) or 0.0),
            best_bid_size=float(_to_float(row.get("up_best_bid_size"), 0.0) or 0.0),
            best_ask_size=float(_to_float(row.get("up_best_ask_size"), 0.0) or 0.0),
            top3_bid_size=float(_to_float(row.get("up_top3_bid_size"), 0.0) or 0.0),
            top3_ask_size=float(_to_float(row.get("up_top3_ask_size"), 0.0) or 0.0),
        ),
        down=OutcomeBook(
            token_id=str(row.get("down_token_id") or "DOWN"),
            best_bid=float(down_bid),
            best_ask=float(down_ask),
            midpoint=float(down_mid),
            spread=float(down_spread),
            topk_imbalance=float(_to_float(row.get("down_topk_imbalance"), 0.0) or 0.0),
            best_bid_size=float(_to_float(row.get("down_best_bid_size"), 0.0) or 0.0),
            best_ask_size=float(_to_float(row.get("down_best_ask_size"), 0.0) or 0.0),
            top3_bid_size=float(_to_float(row.get("down_top3_bid_size"), 0.0) or 0.0),
            top3_ask_size=float(_to_float(row.get("down_top3_ask_size"), 0.0) or 0.0),
        ),
    )


def _taker_fee_map(books: MarketBooks, row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    up_fee = _to_float(row.get("up_taker_fee_bps"))
    down_fee = _to_float(row.get("down_taker_fee_bps"))
    if up_fee is not None:
        out[books.up.token_id] = float(up_fee)
    if down_fee is not None:
        out[books.down.token_id] = float(down_fee)
    return out


def _maker_fill_probability_map(books: MarketBooks, seconds_to_expiry: float, exec_cfg: dict[str, Any]) -> dict[str, float]:
    kwargs = {
        "maker_fill_floor": float(exec_cfg.get("maker_fill_floor", 0.05)),
        "maker_fill_cap": float(exec_cfg.get("maker_fill_cap", 0.9)),
        "maker_fill_base": float(exec_cfg.get("maker_fill_base", 0.78)),
        "maker_fill_spread_penalty": float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        "maker_fill_late_penalty_90": float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        "maker_fill_late_penalty_45": float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    }
    return {
        books.up.token_id: estimate_maker_fill_probability(
            spread=float(books.up.spread),
            seconds_to_expiry=float(seconds_to_expiry),
            **kwargs,
        ),
        books.down.token_id: estimate_maker_fill_probability(
            spread=float(books.down.spread),
            seconds_to_expiry=float(seconds_to_expiry),
            **kwargs,
        ),
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("market_id") or ""),
        str(row.get("created_at") or row.get("ts_utc") or ""),
    )


def _row_quality(row: dict[str, Any]) -> tuple[int, int, int, str]:
    candidate_models = row.get("candidate_models")
    candidate_count = len(candidate_models) if isinstance(candidate_models, dict) else 0
    return (
        candidate_count,
        1 if bool(row.get("trade_has_close")) else 0,
        1 if bool(row.get("is_trainable_v2")) else 0,
        str(row.get("resolved_ts_utc") or row.get("ts_utc") or ""),
    )


def _merge_rows(paths: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            key = _row_key(row)
            prev = by_key.get(key)
            if prev is None or _row_quality(row) > _row_quality(prev):
                by_key[key] = row
    rows = list(by_key.values())
    rows.sort(key=lambda row: str(row.get("created_at") or row.get("resolved_ts_utc") or ""))
    return rows


def _filter_rows(rows: Iterable[dict[str, Any]], clean_only: bool, require_v2: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        side = str(row.get("actual_side", "")).strip().lower()
        if side not in {"up", "down"}:
            continue
        if clean_only and not bool(row.get("is_clean", False)):
            continue
        if require_v2 and not bool(row.get("is_trainable_v2", False)):
            continue
        out.append(row)
    return out


def _candidate_item(row: dict[str, Any], name: str) -> dict[str, Any] | None:
    payload = row.get("candidate_models")
    if not isinstance(payload, dict):
        return None
    item = payload.get(name)
    return item if isinstance(item, dict) else None


def _candidate_prob(row: dict[str, Any], name: str, spot_cfg: SpotWindowPathConfig) -> float | None:
    item = _candidate_item(row, name)
    if item is not None and item.get("p_up") is not None:
        return _clip_prob(item.get("p_up"))

    if name == "proxy_orderbook":
        if row.get("proxy_p_up") is not None:
            return _clip_prob(row.get("proxy_p_up"))
        if str(row.get("model_source", "")).strip() == "proxy_orderbook" and row.get("p_up") is not None:
            return _clip_prob(row.get("p_up"))
        return None

    if name == "spot_window_path":
        return predict_spot_window_path_probability(
            spot_return_bps_from_open=_to_float(row.get("spot_return_bps_from_open")),
            spot_recent_return_1m_bps=_to_float(row.get("spot_recent_return_1m_bps")),
            spot_recent_vol_5m_bps=_to_float(row.get("spot_recent_vol_5m_bps")),
            seconds_to_expiry=_to_float(row.get("seconds_to_expiry"), 300.0),
            cfg=spot_cfg,
        )

    if name == "spot_market_blend":
        if row.get("market_p_up") is None:
            return None
        p_spot = _candidate_prob(row, "spot_window_path", spot_cfg)
        if p_spot is None:
            return None
        return blend_spot_with_market_probability(
            market_p_up=_clip_prob(row.get("market_p_up")),
            spot_p_up=p_spot,
            cfg=spot_cfg,
        )

    if name == "spot_logistic_online":
        if str(row.get("model_source", "")).strip() == "spot_logistic_online" and row.get("p_up") is not None:
            return _clip_prob(row.get("p_up"))
        if row.get("model_p_up") is not None and str(row.get("model_source", "")).strip() == "spot_logistic_online":
            return _clip_prob(row.get("model_p_up"))
        return None

    if name == "spot_logistic_market_blend":
        direct = _candidate_item(row, name)
        if direct is not None and direct.get("p_up") is not None:
            return _clip_prob(direct.get("p_up"))
        return None

    if name == "spot_consensus_blend":
        direct = _candidate_item(row, name)
        if direct is not None and direct.get("p_up") is not None:
            return _clip_prob(direct.get("p_up"))
        return None

    if name == "proxy_logistic_market_blend":
        direct = _candidate_item(row, name)
        if direct is not None and direct.get("p_up") is not None:
            return _clip_prob(direct.get("p_up"))
        return None

    return None


def _candidate_row_result(
    row: dict[str, Any],
    name: str,
    spot_cfg: SpotWindowPathConfig,
    spot_logistic_seed_model: Any,
    causal_spot_logistic_probs: dict[tuple[str, str], float],
    fee_cfg: FeeModelConfig,
    exec_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    p_up = _candidate_prob(row, name, spot_cfg)
    if p_up is None and name == "spot_logistic_online":
        p_up = causal_spot_logistic_probs.get(spot_logistic_row_key(row))
        if p_up is None:
            p_up = predict_spot_logistic_probability(spot_logistic_seed_model, row)
    if p_up is None and name == "spot_logistic_market_blend":
        market_p_up = row.get("market_p_up")
        p_log = causal_spot_logistic_probs.get(spot_logistic_row_key(row))
        if p_log is None:
            p_log = predict_spot_logistic_probability(spot_logistic_seed_model, row)
        if market_p_up is not None and p_log is not None:
            logistic_weight = float(exec_cfg.get("_spot_logistic_market_blend_weight", 0.2))
            p_up = _clip_prob(logistic_weight * p_log + (1.0 - logistic_weight) * _clip_prob(market_p_up))
    if p_up is None and name == "proxy_logistic_market_blend":
        market_p_up = row.get("market_p_up")
        p_proxy = _candidate_prob(row, "proxy_orderbook", spot_cfg)
        p_log = causal_spot_logistic_probs.get(spot_logistic_row_key(row))
        if p_log is None:
            p_log = predict_spot_logistic_probability(spot_logistic_seed_model, row)
        if market_p_up is not None and p_proxy is not None:
            proxy_weight = float(exec_cfg.get("_proxy_logistic_market_blend_proxy_weight", 0.1))
            logistic_weight = float(exec_cfg.get("_proxy_logistic_market_blend_logistic_weight", 0.2))
            proxy_weight = max(0.0, proxy_weight)
            logistic_weight = max(0.0, logistic_weight)
            market_weight = max(0.0, 1.0 - proxy_weight - logistic_weight)
            blend_total = proxy_weight + logistic_weight + market_weight
            if blend_total > 0.0:
                logistic_component = p_log if p_log is not None else _clip_prob(market_p_up)
                p_up = _clip_prob(
                    (
                        proxy_weight * _clip_prob(p_proxy)
                        + logistic_weight * _clip_prob(logistic_component)
                        + market_weight * _clip_prob(market_p_up)
                    )
                    / blend_total
                )
    if p_up is None and name == "spot_consensus_blend":
        p_up = predict_spot_consensus_blend_probability(
            spot_window_path_p_up=_candidate_prob(row, "spot_window_path", spot_cfg),
            spot_logistic_p_up=(
                causal_spot_logistic_probs.get(spot_logistic_row_key(row))
                if causal_spot_logistic_probs.get(spot_logistic_row_key(row)) is not None
                else predict_spot_logistic_probability(spot_logistic_seed_model, row)
            ),
            spot_market_blend_p_up=_candidate_prob(row, "spot_market_blend", spot_cfg),
            market_p_up=_to_float(row.get("market_p_up")),
            cfg=SpotConsensusConfig.from_dict(exec_cfg.get("_spot_consensus_blend_cfg")),
        )
    books = _build_books_from_row(row)
    if p_up is None or books is None:
        return None

    predicted_side = "up" if p_up >= 0.5 else "down"
    actual_side = str(row.get("actual_side", "")).strip().lower()
    correct = predicted_side == actual_side

    allowed_order_types = [
        str(x).strip().lower()
        for x in exec_cfg.get("allowed_order_types", ["maker", "taker"])
        if str(x).strip().lower() in {"maker", "taker"}
    ] or ["maker", "taker"]
    seconds_to_expiry = float(_to_float(row.get("seconds_to_expiry"), 300.0) or 300.0)

    decision = decide_trade(
        market_slug=str(row.get("market_slug") or row.get("market_id") or name),
        books=books,
        p_up=float(p_up),
        min_edge_to_trade=float(exec_cfg.get("min_edge_to_trade", 0.004)),
        min_edge_for_taker=float(exec_cfg.get("min_edge_for_taker", 0.009)),
        max_exposure_usd=float(risk_cfg.get("max_exposure_per_window_usd", 150.0)),
        maker_preference=bool(exec_cfg.get("maker_preference", True)),
        fee_cfg=fee_cfg,
        taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
        maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3.0)),
        taker_fee_bps_by_token=_taker_fee_map(books, row),
        maker_fill_probability=float(exec_cfg.get("maker_fill_probability", 0.65)),
        maker_fill_probability_by_token=_maker_fill_probability_map(books, seconds_to_expiry, exec_cfg),
        maker_ev_advantage_required=float(exec_cfg.get("maker_ev_advantage_required", 0.0005)),
        allowed_order_types=allowed_order_types,
        lock_side_to_prediction=bool(exec_cfg.get("lock_side_to_prediction", True)),
        min_reward_to_risk_ratio=float(exec_cfg.get("min_reward_to_risk_ratio", 0.0)),
    )

    result = {
        "p_up": float(p_up),
        "predicted_side": predicted_side,
        "correct": bool(correct),
        "decision_action": decision.action,
        "decision_reason": decision.reason,
        "decision_best_edge": float(decision.best_edge),
        "decision_order_type": None,
        "decision_expected_edge": None,
        "simulated_selected_side": None,
        "simulated_filled": False,
        "simulated_fill_probability": 0.0,
        "simulated_net_pnl": 0.0,
        "simulated_trade": False,
    }
    if decision.intent is None:
        return result

    result["simulated_trade"] = True
    result["decision_order_type"] = str(decision.intent.order_type).upper()
    result["decision_expected_edge"] = float(decision.intent.expected_edge)
    result["simulated_selected_side"] = "up" if decision.intent.token_id == books.up.token_id else "down"
    created_at = row.get("created_at") or row.get("ts_utc")
    if created_at is None:
        return result
    try:
        created_at_ts = _parse_ts(created_at)
        settled_at_ts = _parse_ts(row.get("resolved_ts_utc") or created_at)
    except Exception:
        return result

    engine = PaperExecutionEngine(
        maker_fill_floor=float(exec_cfg.get("maker_fill_floor", 0.05)),
        maker_fill_cap=float(exec_cfg.get("maker_fill_cap", 0.9)),
        maker_fill_base=float(exec_cfg.get("maker_fill_base", 0.78)),
        maker_fill_spread_penalty=float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        maker_fill_late_penalty_90=float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        maker_fill_late_penalty_45=float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    )
    trade = engine.open_trade(
        intent=decision.intent,
        market_id=str(row.get("market_id") or name),
        created_at=created_at_ts,
        up_token_id=books.up.token_id,
        down_token_id=books.down.token_id,
        spread=float(books.up.spread if decision.intent.token_id == books.up.token_id else books.down.spread),
        seconds_to_expiry=seconds_to_expiry,
        taker_fee_bps=_taker_fee_map(books, row).get(decision.intent.token_id),
    )
    settlement = engine.settle_trade(
        trade=trade,
        settled_at=settled_at_ts,
        actual_side=actual_side,
        fee_cfg=fee_cfg,
        taker_slippage_bps=float(exec_cfg.get("slippage_bps_taker", 12.0)),
        maker_slippage_bps=float(exec_cfg.get("slippage_bps_maker", 3.0)),
    )
    result["simulated_filled"] = bool(settlement.filled)
    result["simulated_fill_probability"] = float(trade.fill_probability)
    result["simulated_net_pnl"] = float(settlement.net_pnl)
    return result


def _primary_actual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(str(row.get("model_source") or "") for row in rows)
    closed = [row for row in rows if bool(row.get("trade_has_close"))]
    pnl = [_to_float(row.get("trade_net_pnl")) for row in closed]
    pnl = [x for x in pnl if x is not None]
    return {
        "model_source_counts": dict(by_source),
        "rows": len(rows),
        "closed_trades": len(closed),
        "net_pnl_sum": float(sum(pnl)) if pnl else 0.0,
        "win_rate": float(sum(1 for x in pnl if x > 0.0) / len(pnl)) if pnl else 0.0,
    }


def _summarize_candidates(per_row: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in CANDIDATES:
        results = [row["candidates"][name] for row in per_row if row["candidates"].get(name) is not None]
        if not results:
            summary[name] = {"rows": 0}
            continue
        p_vals = [float(item["p_up"]) for item in results]
        correct = [1.0 if item["correct"] else 0.0 for item in results]
        y = [1.0 if row["actual_side"] == "up" else 0.0 for row in per_row if row["candidates"].get(name) is not None]
        trade_results = [item for item in results if item["simulated_trade"]]
        filled = [item for item in results if item["simulated_filled"]]
        pnl = [float(item["simulated_net_pnl"]) for item in results]
        filled_pnl = [float(item["simulated_net_pnl"]) for item in filled]
        selected_side_hits = [
            1.0
            for item, row in zip(
                results,
                [row for row in per_row if row["candidates"].get(name) is not None],
                strict=True,
            )
            if item.get("simulated_selected_side") in {"up", "down"} and item.get("simulated_selected_side") == row["actual_side"]
        ]
        selected_side_n = sum(1 for item in results if item.get("simulated_selected_side") in {"up", "down"})
        summary[name] = {
            "rows": len(results),
            "raw_accuracy": float(sum(correct) / len(correct)),
            "brier": float(np.mean((np.asarray(p_vals, dtype=float) - np.asarray(y, dtype=float)) ** 2)),
            "avg_p_up": float(sum(p_vals) / len(p_vals)),
            "trade_coverage": float(len(trade_results) / len(results)),
            "no_trade_rate": float(sum(1 for item in results if not item["simulated_trade"]) / len(results)),
            "trades_taken": len(trade_results),
            "filled_trades": len(filled),
            "fill_rate": float(len(filled) / len(trade_results)) if trade_results else 0.0,
            "filled_win_rate": float(sum(1 for x in filled_pnl if x > 0.0) / len(filled_pnl)) if filled_pnl else 0.0,
            "selected_side_accuracy": float(sum(selected_side_hits) / selected_side_n) if selected_side_n else 0.0,
            "net_pnl_sum": float(sum(pnl)),
            "net_pnl_avg": float(sum(pnl) / len(pnl)) if pnl else 0.0,
            "expected_edge_avg": float(
                sum(float(item["decision_expected_edge"]) for item in trade_results if item["decision_expected_edge"] is not None)
                / max(1, sum(1 for item in trade_results if item["decision_expected_edge"] is not None))
            ) if trade_results else 0.0,
        }
    return summary


def _pairwise(per_row: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1 :]:
            a_rows = 0
            raw_a = raw_b = raw_ties = 0
            exec_a = exec_b = exec_ties = 0
            abstain_a = abstain_b = 0
            for row in per_row:
                ra = row["candidates"].get(a)
                rb = row["candidates"].get(b)
                if ra is None or rb is None:
                    continue
                a_rows += 1
                if ra["correct"] and not rb["correct"]:
                    raw_a += 1
                elif rb["correct"] and not ra["correct"]:
                    raw_b += 1
                else:
                    raw_ties += 1

                pnl_a = float(ra["simulated_net_pnl"])
                pnl_b = float(rb["simulated_net_pnl"])
                if pnl_a > pnl_b + 1e-12:
                    exec_a += 1
                elif pnl_b > pnl_a + 1e-12:
                    exec_b += 1
                else:
                    exec_ties += 1

                if (not ra["simulated_trade"]) and rb["simulated_trade"] and pnl_b < 0.0:
                    abstain_a += 1
                if (not rb["simulated_trade"]) and ra["simulated_trade"] and pnl_a < 0.0:
                    abstain_b += 1
            out[f"{a}__vs__{b}"] = {
                "rows_compared": a_rows,
                "raw": {"a_wins": raw_a, "b_wins": raw_b, "ties": raw_ties},
                "execution": {"a_wins": exec_a, "b_wins": exec_b, "ties": exec_ties},
                "abstain_beats_loss": {"a": abstain_a, "b": abstain_b},
            }
    return out


def _row_preview(per_row: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in per_row[: max(0, limit)]:
        out.append(
            {
                "market_id": row["market_id"],
                "created_at": row["created_at"],
                "actual_side": row["actual_side"],
                "primary_model_source": row["primary_model_source"],
                "primary_trade_has_close": row["primary_trade_has_close"],
                "primary_trade_net_pnl": row["primary_trade_net_pnl"],
                "candidates": row["candidates"],
            }
        )
    return out


def _meta_policy_candidate(
    candidates: dict[str, dict[str, Any]],
    *,
    proxy_midpoint_band: float,
    proxy_edge_margin: float,
    fallback_source: str,
) -> dict[str, Any] | None:
    proxy = candidates.get("proxy_orderbook")
    logistic = candidates.get("spot_logistic_online")
    fallback_name = str(fallback_source or "spot_logistic_online").strip().lower()
    fallback_candidates = {
        "spot_logistic_online": logistic,
        "spot_logistic_market_blend": candidates.get("spot_logistic_market_blend"),
        "proxy_logistic_market_blend": candidates.get("proxy_logistic_market_blend"),
        "proxy_orderbook": proxy,
    }
    fallback = fallback_candidates.get(fallback_name)
    if not isinstance(fallback, dict):
        fallback = logistic or candidates.get("spot_logistic_market_blend") or candidates.get("proxy_logistic_market_blend") or proxy
    if not isinstance(proxy, dict) or not isinstance(logistic, dict):
        if not isinstance(fallback, dict):
            return None
        out = dict(fallback)
        out["meta_source"] = "fallback"
        out["meta_reason"] = "missing_proxy_or_logistic"
        return out

    proxy_selected = str(proxy.get("simulated_selected_side") or "").strip().lower()
    logistic_selected = str(logistic.get("simulated_selected_side") or "").strip().lower()
    proxy_predicted = str(proxy.get("predicted_side") or "").strip().lower()
    proxy_trade = bool(proxy.get("simulated_trade"))
    logistic_trade = bool(logistic.get("simulated_trade"))
    proxy_edge = float(proxy.get("decision_expected_edge") or 0.0)
    logistic_edge = float(logistic.get("decision_expected_edge") or 0.0)
    fallback_edge = float(fallback.get("decision_expected_edge") or logistic_edge or 0.0) if isinstance(fallback, dict) else logistic_edge

    chosen: dict[str, Any]
    source: str
    reason: str
    if (
        proxy_trade
        and logistic_trade
        and proxy_selected in {"up", "down"}
        and proxy_selected == logistic_selected
    ):
        chosen = logistic
        source = "spot_logistic_online"
        reason = "proxy_logistic_agree"
    elif (
        proxy_trade
        and proxy_selected in {"up", "down"}
        and proxy_selected != proxy_predicted
        and (not logistic_trade or proxy_selected != logistic_selected)
        and proxy_edge >= (logistic_edge + float(proxy_edge_margin))
    ):
        chosen = proxy
        source = "proxy_orderbook"
        reason = "proxy_ev_flip"
    elif (
        proxy_trade
        and abs(float(proxy.get("p_up", 0.5)) - 0.5) <= float(proxy_midpoint_band)
        and proxy_edge >= (fallback_edge + float(proxy_edge_margin))
    ):
        chosen = proxy
        source = "proxy_orderbook"
        reason = "proxy_near_midpoint"
    else:
        if not isinstance(fallback, dict):
            return None
        chosen = fallback
        source = "fallback"
        reason = "fallback_candidate"

    out = dict(chosen)
    out["meta_source"] = source
    out["meta_reason"] = reason
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Report same-window live candidate winners across one or more datasets.")
    ap.add_argument("--dataset", nargs="+", required=True, help="One or more dataset jsonl files")
    ap.add_argument("--config", default="configs/default.yaml", help="Config path for model/execution params")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--require-v2", action="store_true")
    ap.add_argument("--preview-rows", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    fee_cfg = FeeModelConfig(**(cfg_raw or {}).get("fees", {}))
    exec_cfg = (cfg_raw or {}).get("execution", {})
    risk_cfg = (cfg_raw or {}).get("risk", {})
    spot_cfg = SpotWindowPathConfig.from_dict((cfg_raw or {}).get("model", {}).get("spot_window_path", {}))
    spot_log_cfg = SpotLogisticConfig.from_dict((cfg_raw or {}).get("model", {}).get("spot_logistic", {}))
    seed_rows_file = (cfg_raw or {}).get("model", {}).get("spot_logistic", {}).get("seed_rows_file")
    seed_rows = _read_jsonl(Path(str(seed_rows_file))) if seed_rows_file else []
    spot_logistic_seed_model = fit_rolling_spot_logistic(seed_rows, spot_log_cfg) if seed_rows else None
    exec_cfg = dict(exec_cfg)
    exec_cfg["_spot_logistic_market_blend_weight"] = float(
        (cfg_raw or {}).get("model", {}).get("spot_logistic_market_blend", {}).get("logistic_weight", 0.2)
    )
    exec_cfg["_proxy_logistic_market_blend_proxy_weight"] = float(
        (cfg_raw or {}).get("model", {}).get("proxy_logistic_market_blend", {}).get("proxy_weight", 0.1)
    )
    exec_cfg["_proxy_logistic_market_blend_logistic_weight"] = float(
        (cfg_raw or {}).get("model", {}).get("proxy_logistic_market_blend", {}).get("logistic_weight", 0.2)
    )
    exec_cfg["_spot_consensus_blend_cfg"] = dict((cfg_raw or {}).get("model", {}).get("spot_consensus_blend", {}))
    meta_cfg = (cfg_raw or {}).get("model", {}).get("proxy_logistic_meta_policy", {})
    proxy_midpoint_band = float(meta_cfg.get("proxy_midpoint_band", 0.06))
    proxy_edge_margin = float(meta_cfg.get("proxy_edge_margin", 0.0))
    fallback_source = str(meta_cfg.get("fallback_source", "spot_logistic_online"))

    rows = _filter_rows(_merge_rows([Path(x) for x in args.dataset]), bool(args.clean_only), bool(args.require_v2))
    causal_spot_logistic_probs = build_causal_spot_logistic_probability_map(rows, spot_log_cfg, seed_rows=seed_rows)
    dataset_paths = [Path(x).resolve() for x in args.dataset]
    seed_path = Path(str(seed_rows_file)).resolve() if seed_rows_file else None
    seed_overlap_detected = bool(seed_path is not None and any(path == seed_path for path in dataset_paths))

    per_row: list[dict[str, Any]] = []
    candidate_presence = Counter()
    for row in rows:
        candidates: dict[str, Any] = {}
        for name in CANDIDATES:
            if name == "proxy_logistic_meta_policy":
                continue
            result = _candidate_row_result(
                row,
                name,
                spot_cfg,
                spot_logistic_seed_model,
                causal_spot_logistic_probs,
                fee_cfg,
                exec_cfg,
                risk_cfg,
            )
            if result is not None:
                candidates[name] = result
                candidate_presence[name] += 1
        meta_result = _meta_policy_candidate(
            candidates,
            proxy_midpoint_band=proxy_midpoint_band,
            proxy_edge_margin=proxy_edge_margin,
            fallback_source=fallback_source,
        )
        if meta_result is not None:
            candidates["proxy_logistic_meta_policy"] = meta_result
            candidate_presence["proxy_logistic_meta_policy"] += 1
        per_row.append(
            {
                "market_id": row.get("market_id"),
                "created_at": row.get("created_at"),
                "actual_side": str(row.get("actual_side", "")).strip().lower(),
                "primary_model_source": row.get("model_source"),
                "primary_trade_has_close": bool(row.get("trade_has_close")),
                "primary_trade_net_pnl": _to_float(row.get("trade_net_pnl"), 0.0),
                "candidates": candidates,
            }
        )

    payload = {
        "datasets": [str(Path(x)) for x in args.dataset],
        "rows": len(rows),
        "spot_logistic_fallback_mode": "causal_asof",
        "spot_logistic_seed_overlap_detected": seed_overlap_detected,
        "candidate_presence": dict(candidate_presence),
        "primary_actual": _primary_actual_summary(rows),
        "candidate_summary": _summarize_candidates(per_row),
        "pairwise": _pairwise(per_row),
        "preview": _row_preview(per_row, int(args.preview_rows)),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
