#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO / "configs" / "default.yaml"
DEFAULT_ROOT = REPO / "outputs" / "monitor_only_ag_shadow"


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config is not a YAML object: {path}")
    return raw


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def _utc_tag() -> str:
    return datetime.now(UTC).strftime("recovery_ag_monitor_%Y%m%d_%H%M%S")


def _force_monitor_safety(cfg: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("app", {})["mode"] = "monitor-5m"
    cfg.setdefault("paths", {})
    cfg["paths"]["outputs_dir"] = str(run_dir)
    cfg["paths"]["logs_dir"] = str(run_dir / "logs")
    cfg["paths"]["reports_dir"] = str(run_dir / "reports")

    cfg.setdefault("execution", {})
    cfg["execution"]["live_trading"] = False
    cfg["execution"]["require_confirm_live"] = True

    cfg.setdefault("monitor", {})
    cfg["monitor"]["predictions_file"] = "predictions_5m.jsonl"
    cfg["monitor"]["outcomes_file"] = "outcomes_5m.jsonl"
    cfg["monitor"]["observations_file"] = "monitor_observations_5m.jsonl"
    cfg["monitor"]["observations_raw_l2_depth"] = 25
    cfg["monitor"]["state_file"] = "monitor_5m_state.json"
    cfg["monitor"]["timing_events_file"] = "monitor_timing_events_5m.jsonl"
    cfg["monitor"]["adaptive_risk_shadow_file"] = "monitor_adaptive_risk_shadow_decisions_5m.jsonl"

    cfg.setdefault("paper", {})
    cfg["paper"]["predictions_file"] = "paper_predictions_5m_DISABLED.jsonl"
    cfg["paper"]["outcomes_file"] = "paper_outcomes_5m_DISABLED.jsonl"
    cfg["paper"]["trades_file"] = "paper_trades_5m_DISABLED.jsonl"
    cfg["paper"]["state_file"] = "paper_5m_state_DISABLED.json"
    cfg["paper"]["adaptive_risk_shadow_file"] = "paper_adaptive_risk_shadow_DISABLED.jsonl"

    cfg.setdefault("risk", {}).setdefault("adaptive_risk", {})
    cfg["risk"]["adaptive_risk"]["enabled"] = True
    cfg["risk"]["adaptive_risk"]["mode"] = "shadow"
    cfg["risk"]["adaptive_risk"].setdefault("monitor_shadow_equity_usd", 100.0)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare isolated recovery monitor-only AG shadow collection config.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    tag = str(args.run_tag or _utc_tag()).strip()
    if not tag:
        raise ValueError("empty run tag")
    root_dir = args.root_dir if args.root_dir.is_absolute() else (REPO / args.root_dir)
    run_dir = root_dir / tag
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    base_config = args.base_config if args.base_config.is_absolute() else (REPO / args.base_config)
    cfg = _force_monitor_safety(_load_yaml(base_config), run_dir=run_dir)
    config_path = run_dir / "monitor_only_ag_config.yaml"
    _write_yaml(config_path, cfg)
    config_sha = _sha256(config_path)
    now = datetime.now(UTC).isoformat()
    git_commit = _git_commit()

    freeze_manifest = {
        "schema": "recovery_ag_monitor_freeze_manifest_v1",
        "created_at_utc": now,
        "freeze_cutoff_utc": now,
        "run_tag": tag,
        "run_dir": str(run_dir),
        "git_commit": git_commit,
        "base_config": str(base_config),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "mode": "monitor-5m",
        "live_trading": False,
        "paper_mode_allowed": False,
        "paper_or_live_allowed": False,
        "thresholds_locked": True,
        "no_retune_rule": "Do not change AG thresholds or gate thresholds after inspecting collected rows; changing config requires a new run/freeze.",
    }
    freeze_path = run_dir / "freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze_manifest, indent=2, sort_keys=True), encoding="utf-8")

    launch_context = {
        "schema": "recovery_ag_monitor_launch_context_v1",
        "created_at_utc": now,
        "mode": "monitor-5m",
        "run_tag": tag,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "freeze_manifest": str(freeze_path),
        "git_commit": git_commit,
        "live_trading": False,
        "paper_mode_allowed": False,
        "expected_files": {
            "predictions": str(run_dir / "predictions_5m.jsonl"),
            "outcomes": str(run_dir / "outcomes_5m.jsonl"),
            "observations": str(run_dir / "monitor_observations_5m.jsonl"),
            "state": str(run_dir / "monitor_5m_state.json"),
            "timing_events": str(run_dir / "monitor_timing_events_5m.jsonl"),
            "adaptive_risk_shadow": str(run_dir / "monitor_adaptive_risk_shadow_decisions_5m.jsonl"),
            "log": str(run_dir / "logs" / "btc_polymarket_bot_monitor.log"),
        },
    }
    launch_path = run_dir / "launch_context.json"
    launch_path.write_text(json.dumps(launch_context, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({**launch_context, "freeze_cutoff_utc": now}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
