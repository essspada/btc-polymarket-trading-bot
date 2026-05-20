from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class TransferredMarkovConfig:
    python_bin: str
    project_root: str
    predictor_module: str
    model_config_path: str
    model_weights_path: str
    output_dir: str
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    candles_limit: int = 14000
    request_limit: int = 1000
    request_timeout_seconds: int = 20
    request_pause_seconds: float = 0.1
    predict_timeout_seconds: int = 120
    hold_weight: float = 0.0
    max_prediction_age_minutes: int = 45
    min_refresh_seconds: int = 30


class TransferredMarkovPredictor:
    def __init__(self, cfg: TransferredMarkovConfig) -> None:
        self.cfg = cfg
        self._last_refresh_ts: float = 0.0
        self._last_p_up: float | None = None

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _parse_iso_utc(value: str) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)

    @staticmethod
    def map_probs_to_up(buy_prob: float, sell_prob: float, hold_prob: float, hold_weight: float = 0.0) -> float:
        buy = max(0.0, float(buy_prob))
        sell = max(0.0, float(sell_prob))
        hold = max(0.0, float(hold_prob))

        hw = min(1.0, max(0.0, float(hold_weight)))
        buy_adj = buy + 0.5 * hw * hold
        sell_adj = sell + 0.5 * hw * hold

        den = buy_adj + sell_adj
        if den <= 0:
            return 0.5

        p_up = buy_adj / den
        return min(0.99, max(0.01, p_up))

    def _fetch_klines(self) -> list[list[Any]]:
        out: list[list[Any]] = []
        end_time: int | None = None

        need = max(500, int(self.cfg.candles_limit))
        per_req = max(10, min(1000, int(self.cfg.request_limit)))

        while len(out) < need:
            params: dict[str, Any] = {
                "symbol": self.cfg.symbol,
                "interval": self.cfg.interval,
                "limit": per_req,
            }
            if end_time is not None:
                params["endTime"] = end_time

            r = requests.get(BINANCE_KLINES_URL, params=params, timeout=int(self.cfg.request_timeout_seconds))
            r.raise_for_status()
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break

            out = batch + out
            oldest = int(batch[0][0])
            end_time = oldest - 1

            if len(batch) < per_req:
                break
            time.sleep(float(self.cfg.request_pause_seconds))

        # de-dup by open timestamp and keep chronological order
        seen = set()
        uniq: list[list[Any]] = []
        for row in out:
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
            uniq.append(row)
        uniq.sort(key=lambda x: int(x[0]))
        return uniq[-need:]

    def _write_raw_csv(self, rows: list[list[Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for row in rows:
                ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC).isoformat()
                w.writerow([ts, row[1], row[2], row[3], row[4], row[5]])

    def _run_external_predictor(self, raw_csv: Path, out_csv: Path) -> None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.python_bin,
            "-m",
            self.cfg.predictor_module,
            "--config",
            self.cfg.model_config_path,
            "--data",
            str(raw_csv),
            "--weights",
            self.cfg.model_weights_path,
            "--out",
            str(out_csv),
            "--batch_size",
            "256",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.cfg.project_root,
                capture_output=True,
                text=True,
                timeout=int(self.cfg.predict_timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"predict_sequence timed out after {self.cfg.predict_timeout_seconds}s"
            ) from e
        
        if proc.returncode != 0:
            raise RuntimeError(
                f"predict_sequence failed rc={proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

    def _read_last_prediction(self, out_csv: Path, now: datetime) -> float:
        if not out_csv.exists():
            raise RuntimeError(f"prediction output not found: {out_csv}")

        last: dict[str, str] | None = None
        with out_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                last = row

        if not last:
            raise RuntimeError("prediction csv is empty")

        ts = self._parse_iso_utc(str(last.get("timestamp", "")))
        age_min = (now - ts).total_seconds() / 60.0
        if age_min > float(self.cfg.max_prediction_age_minutes):
            raise RuntimeError(
                f"stale prediction: age_minutes={age_min:.2f} > max={self.cfg.max_prediction_age_minutes}"
            )

        buy = self._to_float(last.get("buy_prob"), 0.0)
        sell = self._to_float(last.get("sell_prob"), 0.0)
        hold = self._to_float(last.get("hold_prob"), 0.0)
        return self.map_probs_to_up(buy, sell, hold, hold_weight=float(self.cfg.hold_weight))

    def predict_up_probability(self, now: datetime) -> float:
        now_ts = now.timestamp()
        if (
            self._last_p_up is not None
            and (now_ts - self._last_refresh_ts) <= float(self.cfg.min_refresh_seconds)
        ):
            return self._last_p_up

        output_dir = Path(self.cfg.output_dir)
        raw_csv = output_dir / "markov_input_btc_5m.csv"
        out_csv = output_dir / "markov_probs_btc_5m.csv"

        rows = self._fetch_klines()
        if len(rows) < 500:
            raise RuntimeError(f"not enough klines rows={len(rows)}")

        self._write_raw_csv(rows, raw_csv)
        self._run_external_predictor(raw_csv, out_csv)
        p_up = self._read_last_prediction(out_csv, now=now)

        self._last_p_up = p_up
        self._last_refresh_ts = now_ts
        return p_up
