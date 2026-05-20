#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs" / "monitor_only_ag_shadow"
TRUE_STRINGS = {"1", "true", "yes", "on"}
LOG_ALERT_KEYWORDS = (
    "dns",
    "gaierror",
    "temporary failure",
    "name resolution",
    "connection",
    "network",
    "timeout",
    "timed out",
    "max retries",
    "refused",
    "reset by peer",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                rows.append(raw)
    return rows


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_STRINGS


def _nested(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _row_key(row: Mapping[str, Any]) -> str | None:
    for key in ("market_id", "market_slug", "event_slug"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate(row: Mapping[str, Any]) -> bool:
    cash = _float(row.get("adaptive_risk_candidate_cash_usd"))
    action = str(row.get("adaptive_risk_action") or "").strip().lower()
    return cash is not None and cash > 0.0 and action != "no_candidate"


def _latest_run_dir(root: Path = DEFAULT_ROOT) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "freeze_manifest.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _process_rows() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(["ps", "-eo", "pid=,etimes=,cmd="], text=True)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            etimes = int(parts[1])
        except ValueError:
            continue
        rows.append({"pid": pid, "etimes_seconds": etimes, "cmd": parts[2]})
    return rows


def _matching_processes(config_path: Path, process_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = process_rows if process_rows is not None else _process_rows()
    needle = str(config_path)
    out = []
    for row in rows:
        cmd = str(row.get("cmd") or "")
        if needle in cmd and "--mode monitor-5m" in cmd:
            out.append(dict(row))
    return out


def _paper_live_files(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    out: list[str] = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(run_dir))
        name = path.name.lower()
        if name.startswith("paper_") or name.startswith("paper-") or name == "live_order_state.json" or "live_order" in name:
            out.append(rel)
    return sorted(out)


def _mtime_age_seconds(path: Path, now_ts: float) -> float | None:
    if not path.exists():
        return None
    return max(0.0, now_ts - path.stat().st_mtime)


def _recent_log_alerts(path: Path, *, max_lines: int = 200, max_alerts: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except Exception:
        return [{"level": "UNKNOWN", "msg": "failed to read monitor log"}]

    alerts: list[dict[str, Any]] = []
    for line in lines:
        lower = line.lower()
        hit_keyword = any(keyword in lower for keyword in LOG_ALERT_KEYWORDS)
        parsed: dict[str, Any] | None = None
        try:
            raw = json.loads(line)
            parsed = raw if isinstance(raw, dict) else None
        except json.JSONDecodeError:
            parsed = None
        level = str((parsed or {}).get("level") or "").upper()
        if level in {"WARNING", "ERROR", "CRITICAL"} or hit_keyword:
            alerts.append(
                {
                    "ts": (parsed or {}).get("ts"),
                    "level": level or "UNKNOWN",
                    "msg": (parsed or {}).get("msg") or line[:240],
                }
            )
    return alerts[-max_alerts:]


def _network_probe(base_url: Any, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    if not base_url:
        return {"checked": False, "ok": None, "reason": "missing gamma_base_url"}
    url = str(base_url).rstrip("/") + "/"
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return {"checked": True, "ok": False, "stage": "parse", "url": url, "error": "invalid gamma_base_url"}
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.getaddrinfo(host, port)
    except OSError as exc:
        return {"checked": True, "ok": False, "stage": "dns", "url": url, "host": host, "error": str(exc)}

    probe_url = urljoin(url, "markets?limit=1")
    req = Request(probe_url, headers={"User-Agent": "btc-polymarket-monitor-health/1.0"})
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            resp.read(256)
            status_code = getattr(resp, "status", None)
    except HTTPError as exc:
        return {
            "checked": True,
            "ok": 200 <= int(exc.code) < 500,
            "stage": "http",
            "url": probe_url,
            "status_code": int(exc.code),
            "error": str(exc),
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {"checked": True, "ok": False, "stage": "http", "url": probe_url, "error": str(exc)}
    return {"checked": True, "ok": True, "stage": "http", "url": probe_url, "host": host, "status_code": status_code}


def _decision_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ts_utc": row.get("ts_utc") or row.get("created_at"),
        "market": row.get("market_slug") or row.get("event_slug") or row.get("market_id"),
        "action": row.get("adaptive_risk_action"),
        "candidate_cash": row.get("adaptive_risk_candidate_cash_usd"),
        "final_cash": row.get("adaptive_risk_final_cash_usd"),
        "runtime_applied": row.get("adaptive_risk_runtime_applied"),
        "reasons": row.get("adaptive_risk_reason_codes"),
    }


def build_health(
    *,
    run_dir: Path,
    process_rows: list[dict[str, Any]] | None = None,
    stale_log_seconds: float = 900.0,
    now_ts: float | None = None,
    probe_network: bool = False,
    network_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    now_ts = time.time() if now_ts is None else float(now_ts)
    run_dir = run_dir.resolve()
    launch = _load_json(run_dir / "launch_context.json")
    freeze = _load_json(run_dir / "freeze_manifest.json")
    config_path = Path(str(freeze.get("config_path") or launch.get("config_path") or run_dir / "monitor_only_ag_config.yaml"))
    cfg = _load_yaml(config_path)
    expected = launch.get("expected_files") if isinstance(launch.get("expected_files"), Mapping) else {}

    predictions_path = Path(str(expected.get("predictions") or run_dir / "predictions_5m.jsonl"))
    outcomes_path = Path(str(expected.get("outcomes") or run_dir / "outcomes_5m.jsonl"))
    observations_path = Path(str(expected.get("observations") or run_dir / "monitor_observations_5m.jsonl"))
    shadow_path = Path(str(expected.get("adaptive_risk_shadow") or run_dir / "monitor_adaptive_risk_shadow_decisions_5m.jsonl"))
    state_path = Path(str(expected.get("state") or run_dir / "monitor_5m_state.json"))
    log_path = Path(str(expected.get("log") or run_dir / "logs" / "btc_polymarket_bot_monitor.log"))

    predictions = _read_jsonl(predictions_path)
    outcomes = _read_jsonl(outcomes_path)
    observations = _read_jsonl(observations_path)
    shadow = _read_jsonl(shadow_path)
    prediction_keys = {key for row in predictions if (key := _row_key(row)) is not None}
    outcome_keys = {key for row in outcomes if (key := _row_key(row)) is not None}
    candidate_keys = {key for row in [*predictions, *shadow] if (key := _row_key(row)) is not None and _candidate(row)}
    runtime_applied_rows = [row for row in [*predictions, *shadow] if _truthy(row.get("adaptive_risk_runtime_applied"))]
    adaptive_prediction_rows = [row for row in predictions if row.get("adaptive_risk_action") is not None]
    decision_rows = shadow if shadow else adaptive_prediction_rows

    config_hash_now = _sha256(config_path)
    config_hash_frozen = freeze.get("config_sha256")
    hash_ok = bool(config_hash_now and config_hash_frozen and config_hash_now == config_hash_frozen)
    processes = _matching_processes(config_path, process_rows=process_rows)
    paper_live = _paper_live_files(run_dir)
    log_age = _mtime_age_seconds(log_path, now_ts)
    log_alerts = _recent_log_alerts(log_path)
    net_probe = (
        _network_probe(_nested(cfg, "polymarket", "gamma_base_url"), timeout_seconds=network_timeout_seconds)
        if probe_network
        else {"checked": False, "ok": None, "reason": "disabled"}
    )

    unsafe: list[str] = []
    warnings: list[str] = []
    if not processes:
        unsafe.append("monitor process is not alive")
    if _nested(cfg, "app", "mode") != "monitor-5m":
        unsafe.append("config app.mode is not monitor-5m")
    if _nested(cfg, "execution", "live_trading") is not False:
        unsafe.append("config execution.live_trading is not false")
    if _nested(cfg, "risk", "adaptive_risk", "mode") != "shadow":
        unsafe.append("adaptive_risk mode is not shadow")
    if runtime_applied_rows:
        unsafe.append("adaptive_risk_runtime_applied=true appeared")
    if paper_live:
        unsafe.append("paper/live files appeared in run dir")
    if not hash_ok:
        unsafe.append("config hash does not match freeze_manifest")
    if launch.get("paper_mode_allowed") is not False:
        unsafe.append("launch_context paper_mode_allowed is not false")
    if launch.get("live_trading") is not False:
        unsafe.append("launch_context live_trading is not false")

    if predictions_path.exists() and len(predictions) == 0:
        warnings.append("predictions file exists but has zero rows")
    elif not predictions_path.exists():
        warnings.append("predictions file has not appeared yet")
    if outcomes_path.exists() and len(outcomes) == 0:
        warnings.append("outcomes file exists but has zero rows")
    elif not outcomes_path.exists():
        warnings.append("outcomes file has not appeared yet")
    if expected.get("observations"):
        if observations_path.exists() and len(observations) == 0:
            warnings.append("observations file exists but has zero rows")
        elif not observations_path.exists():
            warnings.append("observations file has not appeared yet")
    if shadow_path.exists() and len(shadow) == 0:
        warnings.append("adaptive risk shadow file exists but has zero rows")
    elif not shadow_path.exists():
        warnings.append("adaptive risk shadow file has not appeared yet")
    if len(candidate_keys) == 0:
        warnings.append("no shadow trade candidates yet")
    if log_age is None:
        warnings.append("monitor log file has not appeared yet")
    elif log_age > stale_log_seconds:
        warnings.append(f"monitor log stale for {int(log_age)} seconds")
    if log_alerts:
        warnings.append("recent monitor log contains warning/error/network alerts")
    if net_probe.get("checked") and net_probe.get("ok") is False:
        warnings.append(f"gamma API network probe failed at {net_probe.get('stage')}")

    status = "unsafe" if unsafe else ("warning" if warnings else "healthy")
    return {
        "status": status,
        "run_dir": str(run_dir),
        "process": {
            "alive": bool(processes),
            "matches": processes,
        },
        "counts": {
            "prediction_rows": len(predictions),
            "outcome_rows": len(outcomes),
            "observation_rows": len(observations),
            "resolved_prediction_rows": len(prediction_keys & outcome_keys),
            "shadow_trade_candidates": len(candidate_keys),
            "adaptive_risk_decisions": len(decision_rows),
            "adaptive_risk_prediction_rows": len(adaptive_prediction_rows),
        },
        "safety": {
            "adaptive_risk_runtime_applied_true_rows": len(runtime_applied_rows),
            "paper_live_files": paper_live,
            "config_hash_ok": hash_ok,
            "config_sha256": config_hash_now,
            "frozen_config_sha256": config_hash_frozen,
            "app_mode": _nested(cfg, "app", "mode"),
            "live_trading": _nested(cfg, "execution", "live_trading"),
            "adaptive_risk_mode": _nested(cfg, "risk", "adaptive_risk", "mode"),
            "state_file_exists": state_path.exists(),
            "log_age_seconds": log_age,
            "recent_log_alerts": log_alerts,
            "network_probe": net_probe,
        },
        "latest_adaptive_risk_decisions": [_decision_summary(row) for row in decision_rows[-3:]],
        "unsafe_reasons": unsafe,
        "warnings": warnings,
    }


def _print_text(payload: Mapping[str, Any]) -> None:
    counts = payload.get("counts") if isinstance(payload.get("counts"), Mapping) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), Mapping) else {}
    process = payload.get("process") if isinstance(payload.get("process"), Mapping) else {}
    print(f"status: {payload.get('status')}")
    print(f"run_dir: {payload.get('run_dir')}")
    print(f"process_alive: {process.get('alive')}")
    for item in process.get("matches") or []:
        print(f"process: pid={item.get('pid')} etimes_seconds={item.get('etimes_seconds')}")
    print(f"prediction_rows: {counts.get('prediction_rows')}")
    print(f"outcome_rows: {counts.get('outcome_rows')}")
    print(f"observation_rows: {counts.get('observation_rows')}")
    print(f"resolved_prediction_rows: {counts.get('resolved_prediction_rows')}")
    print(f"shadow_trade_candidates: {counts.get('shadow_trade_candidates')}")
    print(f"adaptive_risk_decisions: {counts.get('adaptive_risk_decisions')}")
    print(f"adaptive_risk_runtime_applied_true_rows: {safety.get('adaptive_risk_runtime_applied_true_rows')}")
    print(f"paper_live_files: {safety.get('paper_live_files') or 'none'}")
    print(f"config_hash_ok: {safety.get('config_hash_ok')}")
    print(f"log_age_seconds: {safety.get('log_age_seconds')}")
    print("network_probe:")
    print("  " + json.dumps(safety.get("network_probe") or {}, ensure_ascii=False, sort_keys=True))
    print("recent_log_alerts:")
    alerts = safety.get("recent_log_alerts") if isinstance(safety.get("recent_log_alerts"), list) else []
    if alerts:
        for item in alerts:
            print("  " + json.dumps(item, ensure_ascii=False, sort_keys=True))
    else:
        print("  none")
    print("last_3_adaptive_risk_decisions:")
    latest = payload.get("latest_adaptive_risk_decisions") if isinstance(payload.get("latest_adaptive_risk_decisions"), list) else []
    if latest:
        for item in latest:
            print("  " + json.dumps(item, ensure_ascii=False, sort_keys=True))
    else:
        print("  none")
    if payload.get("unsafe_reasons"):
        print("unsafe_reasons:")
        for reason in payload.get("unsafe_reasons") or []:
            print(f"  - {reason}")
    if payload.get("warnings"):
        print("warnings:")
        for warning in payload.get("warnings") or []:
            print(f"  - {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only health check for recovery monitor-only AG shadow runs.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory. If omitted, uses latest outputs/monitor_only_ag_shadow run.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of concise text.")
    parser.add_argument("--stale-log-seconds", type=float, default=900.0)
    parser.add_argument("--network-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--skip-network-probe", action="store_true", help="Do not probe gamma API DNS/HTTP reachability.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _latest_run_dir()
        if run_dir is None:
            print("status: unsafe")
            print("unsafe_reasons:")
            print("  - no recovery monitor-only run directory found")
            return 1
    if not run_dir.is_absolute():
        run_dir = (REPO / run_dir).resolve()
    payload = build_health(
        run_dir=run_dir,
        stale_log_seconds=args.stale_log_seconds,
        probe_network=not args.skip_network_probe,
        network_timeout_seconds=args.network_timeout_seconds,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(payload)
    return 2 if payload["status"] == "unsafe" else 0


if __name__ == "__main__":
    raise SystemExit(main())
