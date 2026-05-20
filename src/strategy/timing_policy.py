from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimingPolicyStage:
    delay_seconds: int
    candidate_key: str
    extra_edge: float = 0.0


@dataclass
class DeferredTimingDecision:
    market_id: str
    market_slug: str
    event_slug: str
    series_slug: str
    start_time: str
    end_time: str
    timing_policy_name: str
    next_stage_index: int
    deferred_at: str


def parse_timing_policy_configs(model_cfg: Mapping[str, Any] | None) -> dict[str, tuple[TimingPolicyStage, ...]]:
    defaults: dict[str, tuple[TimingPolicyStage, ...]] = {
        "cons180_then_log240": (
            TimingPolicyStage(delay_seconds=180, candidate_key="spot_consensus_blend", extra_edge=0.10),
            TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
        ),
        "log240_only": (
            TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
        ),
        "log180_then_log240": (
            TimingPolicyStage(delay_seconds=180, candidate_key="spot_logistic_online", extra_edge=0.10),
            TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
        ),
        "path180_then_path240": (
            TimingPolicyStage(delay_seconds=180, candidate_key="spot_window_path", extra_edge=0.10),
            TimingPolicyStage(delay_seconds=240, candidate_key="spot_window_path", extra_edge=0.0),
        ),
    }
    raw_cfg = (model_cfg or {}).get("timing_policies", {}) if isinstance(model_cfg, Mapping) else {}
    parsed = dict(defaults)
    if not isinstance(raw_cfg, Mapping):
        return parsed
    for name_raw, stages_raw in raw_cfg.items():
        name = str(name_raw).strip().lower()
        if not name:
            continue
        stages: list[TimingPolicyStage] = []
        if isinstance(stages_raw, list):
            for item in stages_raw:
                if not isinstance(item, Mapping):
                    continue
                try:
                    delay_seconds = int(item.get("delay_seconds", 0))
                except Exception:
                    continue
                candidate_key = str(item.get("candidate") or item.get("candidate_key") or "").strip()
                if delay_seconds <= 0 or not candidate_key:
                    continue
                try:
                    extra_edge = float(item.get("extra_edge", 0.0))
                except Exception:
                    extra_edge = 0.0
                stages.append(
                    TimingPolicyStage(
                        delay_seconds=delay_seconds,
                        candidate_key=candidate_key,
                        extra_edge=max(0.0, extra_edge),
                    )
                )
        if stages:
            parsed[name] = tuple(sorted(stages, key=lambda x: x.delay_seconds))
    return parsed


def parse_timing_policy_base_sources(model_cfg: Mapping[str, Any] | None) -> dict[str, str]:
    defaults = {
        "cons180_then_log240": "spot_consensus_blend",
        "log240_only": "spot_consensus_blend",
        "log180_then_log240": "spot_consensus_blend",
        "path180_then_path240": "spot_consensus_blend",
    }
    raw_cfg = (model_cfg or {}).get("timing_policy_base_sources", {}) if isinstance(model_cfg, Mapping) else {}
    parsed = dict(defaults)
    if not isinstance(raw_cfg, Mapping):
        return parsed
    for name_raw, value_raw in raw_cfg.items():
        name = str(name_raw).strip().lower()
        value = str(value_raw).strip().lower()
        if name and value:
            parsed[name] = value
    return parsed


def initial_due_stage_index(
    stages: tuple[TimingPolicyStage, ...] | list[TimingPolicyStage],
    *,
    seconds_from_start: float,
) -> int | None:
    due = [idx for idx, stage in enumerate(stages) if float(seconds_from_start) >= float(stage.delay_seconds)]
    if not due:
        return None
    return max(due)


def select_timing_policy_stage(
    *,
    stages: tuple[TimingPolicyStage, ...] | list[TimingPolicyStage],
    seconds_from_start: float,
    candidate_models_payload: Mapping[str, Any] | None,
    min_edge_to_trade: float,
    current_stage_index: int | None = None,
) -> dict[str, Any]:
    stages = tuple(stages)
    if not stages:
        return {"action": "skip", "reason": "no_stages", "next_stage_index": None, "stage_index": None}

    if current_stage_index is None:
        current_stage_index = initial_due_stage_index(stages, seconds_from_start=seconds_from_start)
    if current_stage_index is None:
        return {
            "action": "wait",
            "reason": "before_first_stage",
            "next_stage_index": 0,
            "stage_index": None,
            "stage": None,
            "candidate_item": None,
        }

    idx = max(0, min(int(current_stage_index), len(stages) - 1))
    while idx < len(stages):
        stage = stages[idx]
        if float(seconds_from_start) < float(stage.delay_seconds):
            return {
                "action": "wait",
                "reason": "before_stage",
                "next_stage_index": idx,
                "stage_index": idx,
                "stage": stage,
                "candidate_item": None,
            }

        raw_item = (candidate_models_payload or {}).get(stage.candidate_key)
        item = dict(raw_item) if isinstance(raw_item, Mapping) else None
        if item is None or item.get("p_up") is None:
            idx += 1
            if idx >= len(stages):
                return {
                    "action": "skip",
                    "reason": "no_candidate_probability",
                    "next_stage_index": None,
                    "stage_index": idx - 1,
                    "stage": stage,
                    "candidate_item": None,
                }
            next_stage = stages[idx]
            if float(seconds_from_start) < float(next_stage.delay_seconds):
                return {
                    "action": "defer",
                    "reason": "missing_candidate_wait_next",
                    "next_stage_index": idx,
                    "stage_index": idx - 1,
                    "stage": stage,
                    "candidate_item": None,
                }
            continue

        edge = float(item.get("decision_expected_edge") or 0.0)
        required_edge = float(min_edge_to_trade) + max(0.0, float(stage.extra_edge))
        if str(item.get("decision_action") or "").strip().lower() == "trade" and edge >= required_edge:
            return {
                "action": "finalize",
                "reason": "edge_hit",
                "next_stage_index": None,
                "stage_index": idx,
                "stage": stage,
                "candidate_item": item,
            }

        idx += 1
        if idx >= len(stages):
            return {
                "action": "finalize",
                "reason": "stage_fallback",
                "next_stage_index": None,
                "stage_index": idx - 1,
                "stage": stage,
                "candidate_item": item,
            }
        next_stage = stages[idx]
        if float(seconds_from_start) < float(next_stage.delay_seconds):
            return {
                "action": "defer",
                "reason": "wait_for_next_stage",
                "next_stage_index": idx,
                "stage_index": idx - 1,
                "stage": stage,
                "candidate_item": item,
            }

    return {
        "action": "skip",
        "reason": "no_candidate_probability",
        "next_stage_index": None,
        "stage_index": None,
        "stage": None,
        "candidate_item": None,
    }


def timing_policy_entry_mode(
    *,
    stage: TimingPolicyStage | None,
    current_stage_index: int | None,
    seconds_from_start: float,
    late_start_grace_seconds: float = 0.0,
) -> str | None:
    if stage is None:
        return None
    if current_stage_index is not None:
        return "deferred_resume"
    if float(seconds_from_start) > float(stage.delay_seconds) + max(0.0, float(late_start_grace_seconds)) + 1e-9:
        return "late_fresh_start"
    return "fresh_due"


def next_future_stage_index(
    stages: tuple[TimingPolicyStage, ...] | list[TimingPolicyStage],
    *,
    seconds_from_start: float,
    current_stage_index: int,
) -> int | None:
    for idx in range(max(0, int(current_stage_index) + 1), len(stages)):
        if float(seconds_from_start) < float(stages[idx].delay_seconds):
            return idx
    return None
