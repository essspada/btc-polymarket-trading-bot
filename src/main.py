"""Command-line entrypoint.

Dispatches to one of the four runtime modes (sim / monitor-5m / paper-5m /
backtest-walkforward). All real logic lives in `src.runtime.*` — this file
only parses CLI args and loads config.
"""
from __future__ import annotations

import argparse
import json

from src.runtime.modes.backtest import run_backtest, run_backtest_walkforward
from src.runtime.modes.monitor import run_monitor_5m
from src.runtime.modes.paper import run_paper_5m
from src.runtime.modes.sim import run_sim
from src.utils.config import ensure_dirs, load_app_config, set_deterministic_seed


def main() -> int:
    ap = argparse.ArgumentParser(description="BTC Polymarket bot runner")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--mode",
        choices=["sim", "backtest", "backtest-walkforward", "monitor-5m", "paper-5m"],
        default="sim",
    )
    ap.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required extra acknowledgement for live execution paths.",
    )
    ap.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Optional runtime limit for monitor mode. 0 or omit = run continuously.",
    )
    args = ap.parse_args()

    app_config = load_app_config(args.config)
    cfg = app_config.to_runtime_dict()
    ensure_dirs(cfg)
    set_deterministic_seed(int(cfg["app"]["seed"]))

    if args.mode == "sim":
        summary = run_sim(cfg, confirm_live=args.confirm_live)
    elif args.mode == "monitor-5m":
        summary = run_monitor_5m(cfg, max_runtime_minutes=args.max_runtime_minutes)
    elif args.mode == "paper-5m":
        summary = run_paper_5m(cfg, max_runtime_minutes=args.max_runtime_minutes)
    elif args.mode == "backtest-walkforward":
        summary = run_backtest_walkforward(cfg)
    else:
        summary = run_backtest(cfg)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
