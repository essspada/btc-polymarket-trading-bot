from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.execution.paper_execution import PaperExecutionEngine, estimate_maker_fill_probability
from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.strategy.calibration import clip_prob
from src.strategy.signals import decide_trade


@dataclass
class WalkForwardConfig:
    lookback: int = 240
    min_train_samples: int = 40
    min_edge_to_trade: float = 0.004
    min_edge_for_taker: float = 0.009
    max_exposure_usd: float = 150.0
    maker_preference: bool = True
    maker_fill_probability: float = 0.65
    maker_ev_advantage_required: float = 0.0005
    maker_slippage_bps: float = 3.0
    maker_fill_floor: float = 0.05
    maker_fill_cap: float = 0.9
    maker_fill_base: float = 0.78
    maker_fill_spread_penalty: float = 7.0
    maker_fill_late_penalty_90: float = 0.15
    maker_fill_late_penalty_45: float = 0.10
    require_book_quality: bool = False
    max_outcome_spread: float = 1.0
    allowed_order_types: tuple[str, ...] = ("maker", "taker")


def _label(row: dict[str, Any]) -> int | None:
    side = str(row.get("actual_side", "")).strip().lower()
    if side == "up":
        return 1
    if side == "down":
        return 0
    return None


def _safe_prob(value: Any, default: float = 0.5) -> float:
    try:
        return clip_prob(float(value))
    except Exception:
        return clip_prob(default)


def _feature_vector(row: dict[str, Any]) -> list[float]:
    p_up = _safe_prob(row.get("p_up"), 0.5)
    market_p = _safe_prob(row.get("market_p_up"), 0.5)
    proxy_p = _safe_prob(row.get("proxy_p_up"), p_up)
    model_p = _safe_prob(row.get("model_p_up"), p_up)
    sec_to_exp = float(row.get("seconds_to_expiry", 300.0) or 300.0)
    sec_norm = float(np.clip(sec_to_exp / 300.0, 0.0, 1.0))
    return [p_up, market_p, proxy_p, model_p, sec_norm]


def _fit_platt_like(rows: Sequence[dict[str, Any]]) -> LogisticRegression | None:
    y: list[int] = []
    x: list[list[float]] = []
    for row in rows:
        label = _label(row)
        if label is None:
            continue
        y.append(label)
        x.append(_feature_vector(row))

    if len(y) < 10 or len(set(y)) < 2:
        return None

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=int))
    return model


def _sort_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        for k in ("created_at", "ts_utc", "end_time"):
            if row.get(k):
                return str(row.get(k))
        return ""

    return sorted(list(rows), key=key)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _row_ts(row: dict[str, Any], keys: Sequence[str]) -> datetime | None:
    for key in keys:
        ts = _parse_ts(row.get(key))
        if ts is not None:
            return ts
    return None


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


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


def _maker_fill_probability_map(books: MarketBooks, seconds_to_expiry: float, wf_cfg: WalkForwardConfig) -> dict[str, float]:
    kwargs = {
        "maker_fill_floor": float(wf_cfg.maker_fill_floor),
        "maker_fill_cap": float(wf_cfg.maker_fill_cap),
        "maker_fill_base": float(wf_cfg.maker_fill_base),
        "maker_fill_spread_penalty": float(wf_cfg.maker_fill_spread_penalty),
        "maker_fill_late_penalty_90": float(wf_cfg.maker_fill_late_penalty_90),
        "maker_fill_late_penalty_45": float(wf_cfg.maker_fill_late_penalty_45),
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


def _passes_trading_book_filters(row: dict[str, Any], wf_cfg: WalkForwardConfig) -> bool:
    if bool(wf_cfg.require_book_quality) and row.get("book_quality_ok") is not True:
        return False
    max_spread = float(wf_cfg.max_outcome_spread)
    up_spread = _to_float(row.get("up_spread"))
    down_spread = _to_float(row.get("down_spread"))
    if up_spread is not None and float(up_spread) > max_spread:
        return False
    return down_spread is None or float(down_spread) <= max_spread


def run_walkforward_backtest(
    rows: Iterable[dict[str, Any]],
    wf_cfg: WalkForwardConfig,
    fee_cfg: FeeModelConfig,
    taker_slippage_bps: float,
) -> dict[str, Any]:
    data = _sort_rows(rows)
    valid_rows: list[dict[str, Any]] = []
    for row in data:
        label = _label(row)
        if label is None:
            continue
        books = _build_books_from_row(row)
        if books is None:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return {"error": "no_valid_rows", "rows": 0}

    raw_correct = 0
    cal_correct = 0
    raw_brier = 0.0
    cal_brier = 0.0
    trades = 0
    net_pnl = 0.0
    pnl_rows: list[float] = []
    filled_pnl_rows: list[float] = []
    edge_rows: list[float] = []
    calibration_used = 0
    paper_engine = PaperExecutionEngine(
        maker_fill_floor=float(wf_cfg.maker_fill_floor),
        maker_fill_cap=float(wf_cfg.maker_fill_cap),
        maker_fill_base=float(wf_cfg.maker_fill_base),
        maker_fill_spread_penalty=float(wf_cfg.maker_fill_spread_penalty),
        maker_fill_late_penalty_90=float(wf_cfg.maker_fill_late_penalty_90),
        maker_fill_late_penalty_45=float(wf_cfg.maker_fill_late_penalty_45),
    )
    filled_trades = 0
    order_type_counts: dict[str, int] = {"MAKER": 0, "TAKER": 0}
    trading_rows_considered = 0

    for i, row in enumerate(valid_rows):
        y = _label(row)
        assert y is not None

        p_raw = _safe_prob(row.get("p_up"), 0.5)

        # Strict as-of fitting: train only on rows that were resolved by current prediction time.
        start = max(0, i - int(wf_cfg.lookback)) if wf_cfg.lookback > 0 else 0
        train_rows = valid_rows[start:i]
        current_created_ts = _row_ts(row, ("created_at", "ts_utc", "end_time"))
        if current_created_ts is not None:
            strict_train_rows: list[dict[str, Any]] = []
            for tr in train_rows:
                resolved_ts = _row_ts(tr, ("resolved_ts_utc", "ts_utc", "end_time"))
                if resolved_ts is None:
                    continue
                if resolved_ts <= current_created_ts:
                    strict_train_rows.append(tr)
            train_rows = strict_train_rows

        p_cal = p_raw
        if len(train_rows) >= int(wf_cfg.min_train_samples):
            model = _fit_platt_like(train_rows)
            if model is not None:
                p_cal = float(model.predict_proba(np.asarray([_feature_vector(row)], dtype=float))[0, 1])
                p_cal = clip_prob(p_cal)
                calibration_used += 1

        raw_pred = 1 if p_raw >= 0.5 else 0
        cal_pred = 1 if p_cal >= 0.5 else 0

        raw_correct += int(raw_pred == y)
        cal_correct += int(cal_pred == y)
        raw_brier += (p_raw - y) ** 2
        cal_brier += (p_cal - y) ** 2

        books = _build_books_from_row(row)
        if books is None:
            continue
        if not _passes_trading_book_filters(row, wf_cfg):
            continue
        trading_rows_considered += 1
        seconds_to_expiry = float(_to_float(row.get("seconds_to_expiry"), 300.0) or 300.0)
        decision = decide_trade(
            market_slug=str(row.get("market_slug") or row.get("market_id") or "walkforward"),
            books=books,
            p_up=float(p_cal),
            min_edge_to_trade=float(wf_cfg.min_edge_to_trade),
            min_edge_for_taker=float(wf_cfg.min_edge_for_taker),
            max_exposure_usd=float(wf_cfg.max_exposure_usd),
            maker_preference=bool(wf_cfg.maker_preference),
            fee_cfg=fee_cfg,
            taker_slippage_bps=float(taker_slippage_bps),
            maker_slippage_bps=float(wf_cfg.maker_slippage_bps),
            taker_fee_bps_by_token=_taker_fee_map(books, row),
            maker_fill_probability=float(wf_cfg.maker_fill_probability),
            maker_fill_probability_by_token=_maker_fill_probability_map(books, seconds_to_expiry, wf_cfg),
            maker_ev_advantage_required=float(wf_cfg.maker_ev_advantage_required),
            allowed_order_types=list(wf_cfg.allowed_order_types),
            lock_side_to_prediction=bool(getattr(wf_cfg, "lock_side_to_prediction", True)),
            min_reward_to_risk_ratio=float(getattr(wf_cfg, "min_reward_to_risk_ratio", 0.0)),
        )
        if decision.intent is None:
            continue

        trades += 1
        edge_rows.append(float(decision.best_edge))
        created_at = _row_ts(row, ("created_at", "ts_utc", "end_time")) or datetime.now(UTC)
        ref_spread = books.up.spread if decision.intent.token_id == books.up.token_id else books.down.spread
        trade = paper_engine.open_trade(
            intent=decision.intent,
            market_id=str(row.get("market_id") or f"wf_{i}"),
            created_at=created_at,
            up_token_id=books.up.token_id,
            down_token_id=books.down.token_id,
            spread=float(ref_spread),
            seconds_to_expiry=seconds_to_expiry,
            taker_fee_bps=_taker_fee_map(books, row).get(decision.intent.token_id),
        )
        settlement = paper_engine.settle_trade(
            trade=trade,
            settled_at=_row_ts(row, ("resolved_ts_utc", "ts_utc", "end_time")) or created_at,
            actual_side=str(row.get("actual_side", "")).strip().lower(),
            fee_cfg=fee_cfg,
            taker_slippage_bps=float(taker_slippage_bps),
            maker_slippage_bps=float(wf_cfg.maker_slippage_bps),
        )
        order_type_counts[trade.order_type] = order_type_counts.get(trade.order_type, 0) + 1
        if settlement.filled:
            filled_trades += 1
            filled_pnl_rows.append(float(settlement.net_pnl))
        net_pnl += settlement.net_pnl
        pnl_rows.append(float(settlement.net_pnl))

    n = len(valid_rows)
    mean_pnl = net_pnl / trades if trades > 0 else 0.0
    win_rate = (sum(1 for x in pnl_rows if x > 0) / trades) if trades > 0 else 0.0
    mean_filled_pnl = net_pnl / filled_trades if filled_trades > 0 else 0.0
    filled_win_rate = (sum(1 for x in filled_pnl_rows if x > 0) / filled_trades) if filled_trades > 0 else 0.0

    return {
        "rows_total": len(data),
        "rows_valid": n,
        "calibration_fits_used": calibration_used,
        "classification": {
            "raw_accuracy": raw_correct / n,
            "calibrated_accuracy": cal_correct / n,
            "raw_brier": raw_brier / n,
            "calibrated_brier": cal_brier / n,
        },
        "trading": {
            "trading_rows_considered": trading_rows_considered,
            "trades_taken": trades,
            "filled_trades": filled_trades,
            "fill_rate": (filled_trades / trades) if trades > 0 else 0.0,
            "order_type_counts": order_type_counts,
            "avg_net_edge": (sum(edge_rows) / trades) if trades > 0 else 0.0,
            "net_pnl": net_pnl,
            "mean_pnl": mean_pnl,
            "win_rate": win_rate,
            "mean_filled_pnl": mean_filled_pnl,
            "filled_win_rate": filled_win_rate,
        },
    }
