from __future__ import annotations

import json
from pathlib import Path

from scripts.report_historical_pair_coherence import build_coherence_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_build_coherence_report_counts_mismatches_and_writes_filtered_rows(tmp_path: Path) -> None:
    dataset_a = tmp_path / "a.jsonl"
    dataset_b = tmp_path / "b.jsonl"
    filtered_dir = tmp_path / "filtered"
    _write_jsonl(
        dataset_a,
        [
            {"market_id": "m1", "created_at": "2026-03-01T00:00:00+00:00", "entry_price_sum_raw": 1.0, "up_entry_price_raw": 0.4, "down_entry_price_raw": 0.6},
            {"market_id": "m2", "created_at": "2026-03-01T00:05:00+00:00", "entry_price_sum_raw": 1.02, "up_entry_price_raw": 0.5, "down_entry_price_raw": 0.52},
            {"market_id": "m3", "created_at": "2026-03-02T00:00:00+00:00", "entry_price_sum_raw": 0.90, "up_entry_price_raw": 0.35, "down_entry_price_raw": 0.55},
        ],
    )
    _write_jsonl(
        dataset_b,
        [
            {"market_id": "n1", "created_at": "2026-03-01T00:00:00+00:00", "entry_price_sum_raw": 1.0, "up_entry_price_raw": 0.45, "down_entry_price_raw": 0.55},
            {"market_id": "n2", "created_at": "2026-03-02T00:00:00+00:00", "entry_price_sum_raw": 1.03, "up_entry_price_raw": 0.53, "down_entry_price_raw": 0.50},
        ],
    )

    report = build_coherence_report(
        {"180": dataset_a, "240": dataset_b},
        thresholds=[0.00, 0.02],
        out_filtered_dir=filtered_dir,
    )

    assert report["datasets"]["180"]["sum_stats"]["mismatch_count"] == 2
    assert report["datasets"]["240"]["sum_stats"]["mismatch_count"] == 1
    assert report["filtered_datasets"]["0.00"]["180"]["rows"] == 1
    assert report["filtered_datasets"]["0.02"]["180"]["rows"] == 2
    assert report["filtered_datasets"]["0.02"]["240"]["rows"] == 1
    assert Path(report["filtered_datasets"]["0.02"]["180"]["path"]).exists()
