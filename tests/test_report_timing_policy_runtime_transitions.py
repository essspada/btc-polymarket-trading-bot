from __future__ import annotations

import json
from pathlib import Path

from scripts.report_timing_policy_runtime_transitions import build_transition_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_build_transition_report_summarizes_events_by_reason_and_entry_mode(tmp_path: Path) -> None:
    events_path = tmp_path / "timing_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "event": "defer",
                "market_id": "m1",
                "timing_policy_name": "cons180_then_log240",
                "reason": "wait_for_next_stage",
                "timing_policy_entry_mode": "fresh_due",
            },
            {
                "event": "finalize",
                "market_id": "m1",
                "timing_policy_name": "cons180_then_log240",
                "reason": "edge_hit",
                "timing_policy_stage_delay": 240,
                "timing_policy_entry_mode": "deferred_resume",
            },
            {
                "event": "finalize",
                "market_id": "m2",
                "timing_policy_name": "log180_then_log240",
                "reason": "stage_fallback",
                "timing_policy_stage_delay": 240,
                "timing_policy_entry_mode": "late_fresh_start",
            },
            {
                "event": "skip",
                "market_id": "m3",
                "timing_policy_name": "path180_then_path240",
                "reason": "no_candidate_probability",
            },
        ],
    )

    payload = build_transition_report([("paper", events_path)])

    file_stats = payload["files"]["paper"]
    assert file_stats["event_counts"] == {"defer": 1, "finalize": 2, "skip": 1}
    assert file_stats["finalize_reason_counts"] == {"edge_hit": 1, "stage_fallback": 1}
    assert file_stats["finalize_entry_mode_counts"] == {"deferred_resume": 1, "late_fresh_start": 1}
    assert file_stats["defer_reason_counts"] == {"wait_for_next_stage": 1}
    assert file_stats["skip_reason_counts"] == {"no_candidate_probability": 1}
    assert file_stats["finalize_reason_by_entry_mode"] == {
        "edge_hit": {"deferred_resume": 1},
        "stage_fallback": {"late_fresh_start": 1},
    }
