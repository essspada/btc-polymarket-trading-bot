#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _parse_dataset_specs(values: Iterable[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in values:
        label, sep, path_raw = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"dataset spec must be LABEL=PATH, got: {raw}")
        out[label.strip()] = Path(path_raw.strip()).resolve()
    if not out:
        raise SystemExit("at least one dataset is required")
    return out


def _parse_thresholds(raw: str) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise SystemExit("thresholds must not be empty")
    return values


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _extreme_examples(rows: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    items = [
        {
            "market_id": row.get("market_id"),
            "created_at": row.get("created_at"),
            "sum": float(row["entry_price_sum_raw"]),
            "up": row.get("up_entry_price_raw"),
            "down": row.get("down_entry_price_raw"),
        }
        for row in rows
        if isinstance(row.get("entry_price_sum_raw"), (int, float))
    ]
    items.sort(key=lambda item: float(item["sum"]), reverse=bool(reverse))
    return items[:5]


def _rows_for_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        raw_sum = row.get("entry_price_sum_raw")
        if not isinstance(raw_sum, (int, float)):
            continue
        if abs(float(raw_sum) - 1.0) <= float(threshold) + 1e-12:
            out.append(row)
    return out


def build_coherence_report(
    dataset_paths: dict[str, Path],
    *,
    thresholds: list[float],
    out_filtered_dir: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "thresholds": [float(x) for x in thresholds],
        "datasets": {},
        "filtered_datasets": {},
    }

    for label, path in dataset_paths.items():
        rows = _read_jsonl(path)
        sums = [float(row["entry_price_sum_raw"]) for row in rows if isinstance(row.get("entry_price_sum_raw"), (int, float))]
        mismatches = [value for value in sums if abs(value - 1.0) > 1e-12]
        report["datasets"][str(label)] = {
            "path": str(path),
            "rows": len(rows),
            "sum_stats": {
                "count": len(sums),
                "exact_one_count": len(sums) - len(mismatches),
                "mismatch_count": len(mismatches),
                "mismatch_ratio": (len(mismatches) / len(sums)) if sums else None,
                "min": min(sums) if sums else None,
                "p01": _quantile(sums, 0.01),
                "p05": _quantile(sums, 0.05),
                "median": float(median(sums)) if sums else None,
                "p95": _quantile(sums, 0.95),
                "p99": _quantile(sums, 0.99),
                "max": max(sums) if sums else None,
            },
            "extreme_examples": {
                "lowest": _extreme_examples(rows, reverse=False),
                "highest": _extreme_examples(rows, reverse=True),
            },
        }

        for threshold in thresholds:
            kept = _rows_for_threshold(rows, float(threshold))
            day_count = len({str(row.get("created_at") or "")[:10] for row in kept if str(row.get("created_at") or "")[:10]})
            info: dict[str, Any] = {
                "rows": len(kept),
                "days": day_count,
            }
            if out_filtered_dir is not None:
                out_filtered_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_filtered_dir / f"{label}_coherence_le_{threshold:0.2f}.jsonl"
                with out_path.open("w", encoding="utf-8") as f:
                    for row in kept:
                        f.write(json.dumps(row, ensure_ascii=True) + "\n")
                info["path"] = str(out_path.resolve())
            report["filtered_datasets"].setdefault(f"{threshold:0.2f}", {})[str(label)] = info

    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Report pair-price coherence for historical BTC 5m datasets.")
    ap.add_argument("--datasets", nargs="+", required=True, help="One or more LABEL=PATH specs")
    ap.add_argument("--thresholds", default="0.00,0.01,0.02,0.03")
    ap.add_argument("--out-filtered-dir", default=None, help="Optional directory for filtered datasets per threshold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset_paths = _parse_dataset_specs(list(args.datasets))
    thresholds = _parse_thresholds(str(args.thresholds))
    out_filtered_dir = Path(args.out_filtered_dir).resolve() if args.out_filtered_dir else None
    payload = build_coherence_report(dataset_paths, thresholds=thresholds, out_filtered_dir=out_filtered_dir)

    text = json.dumps(payload, indent=2)
    print(text)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
