from src.strategy.timing_policy import (
    TimingPolicyStage,
    initial_due_stage_index,
    parse_timing_policy_base_sources,
    parse_timing_policy_configs,
    select_timing_policy_stage,
    timing_policy_entry_mode,
)


def test_parse_timing_policy_defaults_include_freeze_set() -> None:
    cfgs = parse_timing_policy_configs({})
    assert tuple(stage.delay_seconds for stage in cfgs["cons180_then_log240"]) == (180, 240)
    assert cfgs["cons180_then_log240"][0].candidate_key == "spot_consensus_blend"
    assert cfgs["log180_then_log240"][0].candidate_key == "spot_logistic_online"
    assert cfgs["path180_then_path240"][1].candidate_key == "spot_window_path"

    base_sources = parse_timing_policy_base_sources({})
    assert base_sources["cons180_then_log240"] == "spot_consensus_blend"


def test_initial_due_stage_index_uses_latest_due_stage_on_fresh_late_start() -> None:
    stages = (
        TimingPolicyStage(delay_seconds=180, candidate_key="spot_consensus_blend", extra_edge=0.10),
        TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
    )
    assert initial_due_stage_index(stages, seconds_from_start=170.0) is None
    assert initial_due_stage_index(stages, seconds_from_start=190.0) == 0
    assert initial_due_stage_index(stages, seconds_from_start=250.0) == 1


def test_select_timing_policy_stage_defers_when_early_edge_is_below_extra_margin() -> None:
    stages = (
        TimingPolicyStage(delay_seconds=180, candidate_key="spot_consensus_blend", extra_edge=0.10),
        TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
    )
    result = select_timing_policy_stage(
        stages=stages,
        seconds_from_start=181.0,
        candidate_models_payload={
            "spot_consensus_blend": {
                "p_up": 0.60,
                "decision_action": "trade",
                "decision_expected_edge": 0.05,
            }
        },
        min_edge_to_trade=0.01,
        current_stage_index=0,
    )
    assert result["action"] == "defer"
    assert result["next_stage_index"] == 1
    assert result["reason"] == "wait_for_next_stage"


def test_select_timing_policy_stage_finalizes_last_stage_when_runtime_starts_late() -> None:
    stages = (
        TimingPolicyStage(delay_seconds=180, candidate_key="spot_consensus_blend", extra_edge=0.10),
        TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
    )
    result = select_timing_policy_stage(
        stages=stages,
        seconds_from_start=250.0,
        candidate_models_payload={
            "spot_logistic_online": {
                "p_up": 0.42,
                "decision_action": "trade",
                "decision_expected_edge": 0.03,
            }
        },
        min_edge_to_trade=0.01,
        current_stage_index=None,
    )
    assert result["action"] == "finalize"
    assert result["reason"] == "edge_hit"
    assert result["stage"].candidate_key == "spot_logistic_online"


def test_select_timing_policy_stage_falls_back_on_final_stage_no_trade() -> None:
    stages = (
        TimingPolicyStage(delay_seconds=180, candidate_key="spot_consensus_blend", extra_edge=0.10),
        TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0),
    )
    result = select_timing_policy_stage(
        stages=stages,
        seconds_from_start=240.0,
        candidate_models_payload={
            "spot_logistic_online": {
                "p_up": 0.51,
                "decision_action": "no_trade",
                "decision_expected_edge": None,
            }
        },
        min_edge_to_trade=0.01,
        current_stage_index=1,
    )
    assert result["action"] == "finalize"
    assert result["reason"] == "stage_fallback"


def test_timing_policy_entry_mode_distinguishes_fresh_deferred_and_late_start() -> None:
    stage = TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0)
    assert timing_policy_entry_mode(stage=stage, current_stage_index=None, seconds_from_start=240.0) == "fresh_due"
    assert timing_policy_entry_mode(stage=stage, current_stage_index=1, seconds_from_start=245.0) == "deferred_resume"
    assert timing_policy_entry_mode(stage=stage, current_stage_index=None, seconds_from_start=250.0) == "late_fresh_start"


def test_timing_policy_entry_mode_respects_late_start_grace_window() -> None:
    stage = TimingPolicyStage(delay_seconds=240, candidate_key="spot_logistic_online", extra_edge=0.0)
    assert (
        timing_policy_entry_mode(
            stage=stage,
            current_stage_index=None,
            seconds_from_start=250.0,
            late_start_grace_seconds=10.0,
        )
        == "fresh_due"
    )
    assert (
        timing_policy_entry_mode(
            stage=stage,
            current_stage_index=None,
            seconds_from_start=251.0,
            late_start_grace_seconds=10.0,
        )
        == "late_fresh_start"
    )
