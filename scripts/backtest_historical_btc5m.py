#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.clients.gamma_client import GammaClient, parse_json_list_field
from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.polymarket.market_discovery import build_btc_5m_event_slug, extract_resolved_outcome_side
from src.strategy.spot_consensus import SpotConsensusConfig, predict_spot_consensus_blend_probability
from src.strategy.spot_logistic import (
    SpotLogisticConfig,
    build_causal_spot_logistic_probability_map,
)
from src.strategy.spot_window_path import (
    SpotWindowPathConfig,
    blend_spot_with_market_probability,
    predict_spot_window_path_probability,
)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class HistoricalMarket:
    market_id: str
    market_slug: str
    question: str
    start_time: datetime
    end_time: datetime
    up_token_id: str
    down_token_id: str
    actual_side: str


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _clip_prob(value: Any) -> float:
    try:
        p = float(value)
    except Exception:
        p = 0.5
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _extract_outcome_tokens(market: dict[str, Any]) -> dict[str, str] | None:
    outcomes = [str(x).strip().lower() for x in parse_json_list_field(market.get("outcomes"))]
    token_ids = [str(x) for x in parse_json_list_field(market.get("clobTokenIds"))]
    if len(outcomes) != 2 or len(token_ids) != 2:
        return None
    mapping = dict(zip(outcomes, token_ids, strict=True))
    up = mapping.get("up") or mapping.get("yes")
    down = mapping.get("down") or mapping.get("no")
    if not up or not down:
        return None
    return {"up": up, "down": down}


def _build_market(event: dict[str, Any], market: dict[str, Any], aligned_start: datetime) -> HistoricalMarket | None:
    tokens = _extract_outcome_tokens(market)
    if not tokens:
        return None
    actual_side = extract_resolved_outcome_side(market)
    if actual_side not in {"up", "down"}:
        return None
    end_time = _parse_ts(market.get("endDate") or event.get("endDate"))
    if end_time is None:
        return None
    return HistoricalMarket(
        market_id=str(market.get("id") or ""),
        market_slug=str(market.get("slug") or ""),
        question=str(market.get("question") or ""),
        start_time=aligned_start,
        end_time=end_time,
        up_token_id=tokens["up"],
        down_token_id=tokens["down"],
        actual_side=actual_side,
    )


def _iter_aligned_windows(start_dt: datetime, end_dt: datetime) -> Iterable[datetime]:
    current = start_dt.astimezone(UTC).replace(second=0, microsecond=0)
    aligned = current - timedelta(minutes=current.minute % 5)
    while aligned < end_dt:
        yield aligned
        aligned += timedelta(minutes=5)


def _fetch_historical_markets(
    gamma: GammaClient,
    *,
    start_dt: datetime,
    end_dt: datetime,
    pause_seconds: float,
    workers: int,
    max_windows: int = 0,
    fetch_retries: int = 2,
    retry_pause_seconds: float = 0.25,
    retry_missing_passes: int = 1,
) -> list[HistoricalMarket]:
    def _fetch_one(window_start: datetime) -> HistoricalMarket | None:
        slug = build_btc_5m_event_slug(window_start)
        attempts = max(1, int(fetch_retries) + 1)
        for attempt in range(attempts):
            try:
                events = gamma.get_events(slug=slug, limit=1)
            except Exception:
                events = []
            if pause_seconds > 0:
                time.sleep(pause_seconds)
            if events:
                event = events[0]
                if str(event.get("seriesSlug") or "") != "btc-up-or-down-5m":
                    return None
                for market in event.get("markets", []):
                    item = _build_market(event, market, aligned_start=window_start)
                    if item is not None:
                        return item
            if attempt + 1 < attempts and retry_pause_seconds > 0:
                time.sleep(float(retry_pause_seconds) * float(attempt + 1))
        return None

    windows = list(_iter_aligned_windows(start_dt, end_dt))
    if max_windows > 0:
        windows = windows[: max_windows]
    out: list[HistoricalMarket] = []
    max_workers = max(1, int(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, window_start): window_start for window_start in windows}
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            item = future.result()
            if item is not None:
                out.append(item)
            if idx % 250 == 0:
                print(json.dumps({"market_scan_progress": idx, "markets_found": len(out)}), flush=True)
    if retry_missing_passes > 0 and len(out) < len(windows):
        found_starts = {item.start_time.isoformat() for item in out}
        missing_windows = [window for window in windows if window.isoformat() not in found_starts]
        for pass_idx in range(int(retry_missing_passes)):
            if not missing_windows:
                break
            recovered: list[HistoricalMarket] = []
            for idx, window_start in enumerate(missing_windows, start=1):
                item = _fetch_one(window_start)
                if item is not None:
                    recovered.append(item)
                if idx % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "market_retry_pass": pass_idx + 1,
                                "retry_progress": idx,
                                "retry_total": len(missing_windows),
                                "recovered": len(recovered),
                            }
                        ),
                        flush=True,
                    )
            if recovered:
                out.extend(recovered)
                found_starts.update(item.start_time.isoformat() for item in recovered)
            missing_windows = [window for window in missing_windows if window.isoformat() not in found_starts]
            print(
                json.dumps(
                    {
                        "market_retry_pass": pass_idx + 1,
                        "remaining_missing": len(missing_windows),
                        "total_markets_found": len(out),
                    }
                ),
                flush=True,
            )
    out.sort(key=lambda item: item.start_time)
    return out


def _fetch_binance_klines_1m(symbol: str, start_dt: datetime, end_dt: datetime, timeout: int, pause_seconds: float) -> list[list[Any]]:
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    out: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        payload = r.json()
        rows = payload if isinstance(payload, list) else []
        if not rows:
            break
        out.extend(rows)
        next_open = int(rows[-1][0]) + 60_000
        if next_open <= cursor:
            break
        cursor = next_open
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return out


def _price_at_or_before(history: list[dict[str, Any]], target_ts: int) -> float | None:
    chosen: float | None = None
    for point in history:
        ts = int(point.get("t", -1))
        if ts <= target_ts:
            chosen = _to_float(point.get("p"))
        else:
            break
    return chosen


def _normalize_pair(up_price: float | None, down_price: float | None) -> tuple[float | None, float | None, float | None]:
    if up_price is None or down_price is None:
        return None, None, None
    total = up_price + down_price
    if total <= 0:
        return None, None, None
    up_norm = float(up_price / total)
    down_norm = float(down_price / total)
    return up_norm, down_norm, float(total)


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


def _last_completed_candle(rows: list[list[Any]], entry_ts: datetime) -> list[Any] | None:
    target_ms = int(entry_ts.astimezone(UTC).timestamp() * 1000)
    chosen: list[Any] | None = None
    for row in rows:
        try:
            open_ms = int(row[0])
        except Exception:
            continue
        close_ms = open_ms + 60_000
        if close_ms <= target_ms:
            chosen = row
        else:
            break
    return chosen


def _window_open_reference(window_start: datetime, rows: list[list[Any]]) -> float | None:
    target_ms = int(window_start.astimezone(UTC).timestamp() * 1000)
    chosen: list[Any] | None = None
    for row in rows:
        try:
            open_ms = int(row[0])
        except Exception:
            continue
        if open_ms <= target_ms:
            chosen = row
        if open_ms == target_ms:
            break
    if chosen is None:
        return None
    return _to_float(chosen[1])


def _minute_return_bps(row: list[Any] | None) -> float | None:
    if row is None:
        return None
    open_px = _to_float(row[1])
    close_px = _to_float(row[4])
    if open_px is None or close_px is None or open_px <= 0:
        return None
    return float(10_000.0 * (close_px - open_px) / open_px)


def _recent_vol_bps(rows: list[list[Any]], entry_ts: datetime) -> float | None:
    target_ms = int(entry_ts.astimezone(UTC).timestamp() * 1000)
    recent: list[list[Any]] = []
    for row in rows:
        try:
            open_ms = int(row[0])
        except Exception:
            continue
        close_ms = open_ms + 60_000
        if close_ms <= target_ms:
            recent.append(row)
    recent = recent[-5:]
    returns: list[float] = []
    for row in recent:
        value = _minute_return_bps(row)
        if value is not None:
            returns.append(value)
    if len(returns) < 2:
        return None
    arr = np.asarray(returns, dtype=float)
    return float(arr.std(ddof=0))


def _spot_context_from_klines(rows: list[list[Any]], *, window_start: datetime, entry_ts: datetime) -> dict[str, Any] | None:
    window_open = _window_open_reference(window_start, rows)
    last_completed = _last_completed_candle(rows, entry_ts)
    if window_open is None or last_completed is None:
        return None
    spot_now = _to_float(last_completed[4])
    if spot_now is None or window_open <= 0:
        return None
    return_1m = _minute_return_bps(last_completed)
    vol_5m = _recent_vol_bps(rows, entry_ts)
    return {
        "spot_price_now": float(spot_now),
        "spot_window_open_price": float(window_open),
        "spot_return_bps_from_open": float(10_000.0 * (spot_now - window_open) / window_open),
        "spot_recent_return_1m_bps": float(return_1m or 0.0),
        "spot_recent_vol_5m_bps": float(vol_5m or 0.0),
    }


def _trade_policy_metrics(rows: list[dict[str, Any]], prob_fn: Any, fee_cfg: FeeModelConfig, slippage_bps: float, min_edge_to_trade: float) -> dict[str, Any]:
    usable = 0
    correct = 0
    p_vals: list[float] = []
    y_vals: list[float] = []
    trades = 0
    trade_correct = 0
    pnl_rows: list[float] = []
    edges: list[float] = []
    for row in rows:
        p_up = prob_fn(row)
        if p_up is None:
            continue
        up_price = _to_float(row.get("up_entry_price"))
        down_price = _to_float(row.get("down_entry_price"))
        if up_price is None or down_price is None:
            continue
        usable += 1
        actual_up = 1.0 if row.get("actual_side") == "up" else 0.0
        p_up = _clip_prob(p_up)
        pred_side = "up" if p_up >= 0.5 else "down"
        if pred_side == row.get("actual_side"):
            correct += 1
        p_vals.append(p_up)
        y_vals.append(actual_up)

        p_down = 1.0 - p_up
        up_fee = compute_trade_fee(price=up_price, size=1.0, is_taker=True, cfg=fee_cfg)
        down_fee = compute_trade_fee(price=down_price, size=1.0, is_taker=True, cfg=fee_cfg)
        up_slip = up_price * (slippage_bps / 10_000.0)
        down_slip = down_price * (slippage_bps / 10_000.0)
        edge_up = p_up - up_price - up_fee - up_slip
        edge_down = p_down - down_price - down_fee - down_slip
        best_edge = max(edge_up, edge_down)
        if best_edge < min_edge_to_trade:
            continue
        side = "up" if edge_up >= edge_down else "down"
        trades += 1
        edges.append(float(best_edge))
        if side == row.get("actual_side"):
            trade_correct += 1
        if side == "up":
            pnl = (1.0 - up_price) - up_fee - up_slip if row.get("actual_side") == "up" else (-up_price) - up_fee - up_slip
        else:
            pnl = (1.0 - down_price) - down_fee - down_slip if row.get("actual_side") == "down" else (-down_price) - down_fee - down_slip
        pnl_rows.append(float(pnl))

    return {
        "rows": usable,
        "raw_accuracy": float(correct / usable) if usable else 0.0,
        "brier": float(np.mean((np.asarray(p_vals, dtype=float) - np.asarray(y_vals, dtype=float)) ** 2)) if p_vals else 0.0,
        "trades_taken": trades,
        "trade_coverage": float(trades / usable) if usable else 0.0,
        "selected_side_accuracy": float(trade_correct / trades) if trades else 0.0,
        "net_pnl_sum": float(sum(pnl_rows)) if pnl_rows else 0.0,
        "net_pnl_avg": float(sum(pnl_rows) / len(pnl_rows)) if pnl_rows else 0.0,
        "win_rate": float(sum(1 for x in pnl_rows if x > 0.0) / len(pnl_rows)) if pnl_rows else 0.0,
        "expected_edge_avg": float(sum(edges) / len(edges)) if edges else 0.0,
    }


def _daily_trade_policy_metrics(
    rows: list[dict[str, Any]],
    prob_fn: Any,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    min_edge_to_trade: float,
) -> dict[str, Any]:
    daily: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        created_at = _parse_ts(row.get("created_at"))
        if created_at is None:
            continue
        key = created_at.date().isoformat()
        daily.setdefault(key, []).append(row)
    pnl_by_day: list[float] = []
    per_day: dict[str, dict[str, Any]] = {}
    for day_key, day_rows in sorted(daily.items()):
        summary = _trade_policy_metrics(day_rows, prob_fn, fee_cfg, slippage_bps, min_edge_to_trade)
        per_day[day_key] = {
            "rows": int(summary.get("rows", 0)),
            "raw_accuracy": float(summary.get("raw_accuracy", 0.0)),
            "trades_taken": int(summary.get("trades_taken", 0)),
            "selected_side_accuracy": float(summary.get("selected_side_accuracy", 0.0)),
            "net_pnl_sum": float(summary.get("net_pnl_sum", 0.0)),
        }
        pnl_by_day.append(float(summary.get("net_pnl_sum", 0.0)))
    if not pnl_by_day:
        return {"days": 0, "positive_days": 0, "daily": {}}
    arr = np.asarray(pnl_by_day, dtype=float)
    return {
        "days": int(len(pnl_by_day)),
        "positive_days": int(sum(1 for x in pnl_by_day if x > 0.0)),
        "median_daily_pnl": float(np.median(arr)),
        "min_daily_pnl": float(arr.min()),
        "max_daily_pnl": float(arr.max()),
        "daily": per_day,
    }


def _build_market_row(
    clob: ClobClient,
    market: HistoricalMarket,
    *,
    entry_delay_seconds: int,
    klines: list[list[Any]],
    spot_cfg: SpotWindowPathConfig,
    pause_seconds: float,
) -> dict[str, Any] | None:
    entry_ts = market.start_time + timedelta(seconds=entry_delay_seconds)
    history_start_ts = int((entry_ts - timedelta(hours=1)).timestamp())
    history_end_ts = int(entry_ts.timestamp())
    try:
        up_hist_payload = clob.get_prices_history(
            market.up_token_id,
            interval=None,
            fidelity=1,
            start_ts=history_start_ts,
            end_ts=history_end_ts,
        )
        down_hist_payload = clob.get_prices_history(
            market.down_token_id,
            interval=None,
            fidelity=1,
            start_ts=history_start_ts,
            end_ts=history_end_ts,
        )
        up_hist = up_hist_payload.get("history", []) if isinstance(up_hist_payload, dict) else []
        down_hist = down_hist_payload.get("history", []) if isinstance(down_hist_payload, dict) else []
    except Exception as exc:
        return {"status": "missing_price", "market_id": market.market_id, "error": str(exc)}
    if pause_seconds > 0:
        time.sleep(pause_seconds)

    up_price_raw = _price_at_or_before(up_hist, int(entry_ts.timestamp()))
    down_price_raw = _price_at_or_before(down_hist, int(entry_ts.timestamp()))
    up_prob_norm, down_prob_norm, price_sum = _normalize_pair(up_price_raw, down_price_raw)
    if up_price_raw is None or down_price_raw is None or up_prob_norm is None or down_prob_norm is None:
        return {"status": "missing_price", "market_id": market.market_id}

    spot = _spot_context_from_klines(klines, window_start=market.start_time, entry_ts=entry_ts)
    if spot is None:
        return {"status": "missing_spot", "market_id": market.market_id}

    row = {
        "status": "ok",
        "market_id": market.market_id,
        "market_slug": market.market_slug,
        "question": market.question,
        "start_time": market.start_time.isoformat(),
        "end_time": market.end_time.isoformat(),
        "created_at": entry_ts.isoformat(),
        "entry_delay_seconds": entry_delay_seconds,
        "seconds_to_expiry": float(max(0, 300 - entry_delay_seconds)),
        "actual_side": market.actual_side,
        "up_token_id": market.up_token_id,
        "down_token_id": market.down_token_id,
        "up_entry_price_raw": up_price_raw,
        "down_entry_price_raw": down_price_raw,
        # Execution cost should use the actual token price, not the normalized pair probability.
        "up_entry_price": up_price_raw,
        "down_entry_price": down_price_raw,
        "up_entry_price_norm": up_prob_norm,
        "down_entry_price_norm": down_prob_norm,
        "entry_price_sum_raw": price_sum,
        "market_p_up": up_prob_norm,
        "proxy_p_up": up_prob_norm,
        **spot,
    }
    row["spot_window_path_p_up"] = predict_spot_window_path_probability(
        spot_return_bps_from_open=_to_float(row.get("spot_return_bps_from_open")),
        spot_recent_return_1m_bps=_to_float(row.get("spot_recent_return_1m_bps")),
        spot_recent_vol_5m_bps=_to_float(row.get("spot_recent_vol_5m_bps")),
        seconds_to_expiry=_to_float(row.get("seconds_to_expiry"), 240.0) or 240.0,
        cfg=spot_cfg,
    )
    if row["spot_window_path_p_up"] is not None:
        row["spot_market_blend_p_up"] = blend_spot_with_market_probability(
            market_p_up=float(row["market_p_up"]),
            spot_p_up=float(row["spot_window_path_p_up"]),
            cfg=spot_cfg,
        )
    else:
        row["spot_market_blend_p_up"] = None
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and evaluate a longer-horizon historical BTC 5m backtest using Gamma + CLOB prices-history + Binance spot.")
    ap.add_argument("--start-date", required=True, help="Inclusive UTC date, YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="Exclusive UTC date, YYYY-MM-DD")
    ap.add_argument("--entry-delay-seconds", type=int, default=60, help="Decision timestamp inside each 5m window")
    ap.add_argument("--max-windows", type=int, default=0, help="Optional cap for smoke runs")
    ap.add_argument("--gamma-pause-seconds", type=float, default=0.01)
    ap.add_argument("--gamma-fetch-retries", type=int, default=2)
    ap.add_argument("--gamma-retry-pause-seconds", type=float, default=0.25)
    ap.add_argument("--gamma-retry-missing-passes", type=int, default=1)
    ap.add_argument("--clob-pause-seconds", type=float, default=0.01)
    ap.add_argument("--binance-pause-seconds", type=float, default=0.02)
    ap.add_argument("--scan-workers", type=int, default=16)
    ap.add_argument("--price-workers", type=int, default=16)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--in-dataset", default=None, help="Optional existing dataset jsonl to re-evaluate without refetching")
    ap.add_argument("--out-dataset", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    gamma = GammaClient(str((cfg_raw or {}).get("polymarket", {}).get("gamma_base_url", "https://gamma-api.polymarket.com")))
    clob = ClobClient(str((cfg_raw or {}).get("polymarket", {}).get("clob_base_url", "https://clob.polymarket.com")))
    fee_cfg = FeeModelConfig(**(cfg_raw or {}).get("fees", {}))
    exec_cfg = (cfg_raw or {}).get("execution", {})
    model_cfg = (cfg_raw or {}).get("model", {})
    spot_cfg = SpotWindowPathConfig.from_dict(model_cfg.get("spot_window_path", {}))
    spot_log_cfg = SpotLogisticConfig.from_dict(model_cfg.get("spot_logistic", {}))
    spot_consensus_cfg = SpotConsensusConfig.from_dict(model_cfg.get("spot_consensus_blend", {}))

    start_dt = datetime.combine(_parse_date(args.start_date), dt_time.min, tzinfo=UTC)
    end_dt = datetime.combine(_parse_date(args.end_date), dt_time.min, tzinfo=UTC)
    entry_delay_seconds = max(1, int(args.entry_delay_seconds))

    rows: list[dict[str, Any]] = []
    missing_price = 0
    missing_spot = 0
    markets_found = 0
    if args.in_dataset:
        rows = _read_jsonl(Path(args.in_dataset))
        markets_found = len(rows)
    else:
        markets = _fetch_historical_markets(
            gamma,
            start_dt=start_dt,
            end_dt=end_dt,
            pause_seconds=float(args.gamma_pause_seconds),
            workers=int(args.scan_workers),
            max_windows=int(args.max_windows),
            fetch_retries=int(args.gamma_fetch_retries),
            retry_pause_seconds=float(args.gamma_retry_pause_seconds),
            retry_missing_passes=int(args.gamma_retry_missing_passes),
        )
        markets_found = len(markets)
        print(json.dumps({"historical_markets_found": len(markets)}), flush=True)

        klines = _fetch_binance_klines_1m(
            symbol=str(args.symbol),
            start_dt=start_dt - timedelta(minutes=10),
            end_dt=end_dt,
            timeout=20,
            pause_seconds=float(args.binance_pause_seconds),
        )
        print(json.dumps({"binance_1m_klines": len(klines)}), flush=True)

        max_price_workers = max(1, int(args.price_workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_price_workers) as executor:
            futures = {
                executor.submit(
                    _build_market_row,
                    clob,
                    market,
                    entry_delay_seconds=entry_delay_seconds,
                    klines=klines,
                    spot_cfg=spot_cfg,
                    pause_seconds=float(args.clob_pause_seconds),
                ): market.market_id
                for market in markets
            }
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                item = future.result()
                if item is None:
                    continue
                status = str(item.get("status") or "")
                if status == "missing_price":
                    missing_price += 1
                elif status == "missing_spot":
                    missing_spot += 1
                elif status == "ok":
                    item.pop("status", None)
                    rows.append(item)
                if idx % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "dataset_progress": idx,
                                "rows_built": len(rows),
                                "missing_price": missing_price,
                                "missing_spot": missing_spot,
                            }
                        ),
                        flush=True,
                    )

    rows.sort(key=lambda row: str(row.get("created_at") or ""))
    need_logistic = any(row.get("spot_logistic_online_p_up") is None for row in rows)
    logistic_map = build_causal_spot_logistic_probability_map(rows, spot_log_cfg) if need_logistic else {}
    for row in rows:
        if row.get("spot_logistic_online_p_up") is None:
            row["spot_logistic_online_p_up"] = logistic_map.get((str(row.get("market_id") or ""), str(row.get("created_at") or "")))
        row["spot_consensus_blend_p_up"] = predict_spot_consensus_blend_probability(
            spot_window_path_p_up=_to_float(row.get("spot_window_path_p_up")),
            spot_logistic_p_up=_to_float(row.get("spot_logistic_online_p_up")),
            spot_market_blend_p_up=_to_float(row.get("spot_market_blend_p_up")),
            market_p_up=_to_float(row.get("market_p_up")),
            cfg=spot_consensus_cfg,
        )

    min_edge_to_trade = float(exec_cfg.get("min_edge_to_trade", 0.004))
    slippage_bps_taker = float(exec_cfg.get("slippage_bps_taker", 12.0))
    candidate_defs = {
        "market_price_baseline": lambda row: row.get("market_p_up"),
        "spot_window_path": lambda row: row.get("spot_window_path_p_up"),
        "spot_market_blend": lambda row: row.get("spot_market_blend_p_up"),
        "spot_logistic_online": lambda row: row.get("spot_logistic_online_p_up"),
        "spot_consensus_blend": lambda row: row.get("spot_consensus_blend_p_up"),
    }

    candidate_summary: dict[str, Any] = {}
    for name, prob_fn in candidate_defs.items():
        metrics = _trade_policy_metrics(rows, prob_fn, fee_cfg, slippage_bps_taker, min_edge_to_trade)
        metrics.update(_daily_trade_policy_metrics(rows, prob_fn, fee_cfg, slippage_bps_taker, min_edge_to_trade))
        candidate_summary[name] = metrics

    summary = {
        "config": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "entry_delay_seconds": entry_delay_seconds,
            "symbol": args.symbol,
            "max_windows": int(args.max_windows),
            "in_dataset": str(args.in_dataset) if args.in_dataset else None,
        },
        "scan": {
            "markets_found": markets_found,
            "rows_built": len(rows),
            "missing_price_rows": missing_price,
            "missing_spot_rows": missing_spot,
        },
        "candidates": candidate_summary,
    }

    dataset_path = Path(args.out_dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary_path = Path(args.out_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
