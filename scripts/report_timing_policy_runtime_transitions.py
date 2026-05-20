#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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


def _parse_specs(values: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in values:
        label, sep, path_raw = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"event spec must be LABEL=PATH, got: {raw}")
        out.append((label.strip(), Path(path_raw.strip()).resolve()))
    if not out:
        raise SystemExit("at least one event file is required")
    return out


def build_transition_report(specs: list[tuple[str, Path]]) -> dict[str, Any]:
    total_event_counts = Counter()
    total_finalize_reason_counts = Counter()
    total_finalize_delay_counts = Counter()
    total_finalize_entry_mode_counts = Counter()
    total_defer_reason_counts = Counter()
    total_skip_reason_counts = Counter()
    all_markets: set[str] = set()
    files: dict[str, Any] = {}

    for label, path in specs:
        rows = _read_jsonl(path)
        event_counts = Counter()
        policy_counts = Counter()
        finalize_reason_counts = Counter()
        finalize_delay_counts = Counter()
        finalize_entry_mode_counts = Counter()
        defer_reason_counts = Counter()
        skip_reason_counts = Counter()
        finalize_reason_by_entry_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
        markets: set[str] = set()

        for row in rows:
            event = str(row.get("event") or "").strip().lower()
            reason = str(row.get("reason") or "").strip().lower()
            policy = str(row.get("timing_policy_name") or "").strip().lower()
            entry_mode = str(row.get("timing_policy_entry_mode") or "").strip().lower()
            delay = row.get("timing_policy_stage_delay")
            market_id = str(row.get("market_id") or "").strip()

            if market_id:
                markets.add(market_id)
                all_markets.add(market_id)
            if event:
                event_counts[event] += 1
                total_event_counts[event] += 1
            if policy:
                policy_counts[policy] += 1
            if event == "finalize":
                key_delay = str(delay) if delay is not None else "none"
                key_mode = entry_mode or "unknown"
                if reason:
                    finalize_reason_counts[reason] += 1
                    total_finalize_reason_counts[reason] += 1
                    finalize_reason_by_entry_mode[reason][key_mode] += 1
                finalize_delay_counts[key_delay] += 1
                total_finalize_delay_counts[key_delay] += 1
                finalize_entry_mode_counts[key_mode] += 1
                total_finalize_entry_mode_counts[key_mode] += 1
            elif event == "defer" and reason:
                defer_reason_counts[reason] += 1
                total_defer_reason_counts[reason] += 1
            elif event == "skip" and reason:
                skip_reason_counts[reason] += 1
                total_skip_reason_counts[reason] += 1

        files[label] = {
            "path": str(path),
            "events": len(rows),
            "distinct_markets": len(markets),
            "event_counts": dict(event_counts),
            "policy_counts": dict(policy_counts),
            "finalize_reason_counts": dict(finalize_reason_counts),
            "finalize_delay_counts": dict(finalize_delay_counts),
            "finalize_entry_mode_counts": dict(finalize_entry_mode_counts),
            "defer_reason_counts": dict(defer_reason_counts),
            "skip_reason_counts": dict(skip_reason_counts),
            "finalize_reason_by_entry_mode": {
                reason: dict(counts) for reason, counts in finalize_reason_by_entry_mode.items()
            },
        }

    return {
        "files": files,
        "totals": {
            "events": sum(item["events"] for item in files.values()),
            "distinct_markets": len(all_markets),
            "event_counts": dict(total_event_counts),
            "finalize_reason_counts": dict(total_finalize_reason_counts),
            "finalize_delay_counts": dict(total_finalize_delay_counts),
            "finalize_entry_mode_counts": dict(total_finalize_entry_mode_counts),
            "defer_reason_counts": dict(total_defer_reason_counts),
            "skip_reason_counts": dict(total_skip_reason_counts),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize runtime timing-policy transition events.")
    ap.add_argument("--events", nargs="+", required=True, help="One or more LABEL=PATH specs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = build_transition_report(_parse_specs(list(args.events)))
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
