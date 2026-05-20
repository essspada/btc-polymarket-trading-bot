from src.strategy.reliability_memory import build_reliability_memory, extract_regime_tags, score_row_reliability


def _candidate_disagreement_row(pnl: float, cash: float = 10.0) -> dict:
    return {
        "predicted_side": "up",
        "actual_side": "down" if pnl < 0 else "up",
        "confidence": 0.91,
        "decision_action": "trade",
        "decision_order_type": "MAKER",
        "decision_best_edge": 0.08,
        "timing_policy_reason": "edge_hit",
        "timing_policy_stage_delay": 240,
        "spot_recent_vol_5m_bps": 3.0,
        "up_best_bid": 0.62,
        "trade_filled": True,
        "trade_net_pnl": pnl,
        "sizing_cap_capped_cash_required": cash,
        "candidate_models": {
            "a": {"p_up": 0.20},
            "b": {"p_up": 0.30},
            "c": {"p_up": 0.40},
        },
    }


def test_extract_regime_tags_uses_low_cardinality_bins() -> None:
    tags = extract_regime_tags(_candidate_disagreement_row(-4.0))
    assert tags["side"] == "up"
    assert tags["confidence_bin"] == "0.90-0.95"
    assert tags["edge_bin"] == "0.07-0.15"
    assert tags["timing_reason"] == "edge_hit"
    assert tags["vol_bin"] == "2-5bps"


def test_reliability_memory_shrinks_sparse_loss() -> None:
    memory = build_reliability_memory(
        [_candidate_disagreement_row(-10.0)],
        prior_filled=9.0,
        min_filled_for_label=3,
        toxic_roi_threshold=-0.05,
    )
    candidate_entries = [
        item
        for item in memory["entries"]
        if item["expert"] == "candidate_agreement_expert" and item["event"] == "veto" and item["scope"] == "expert"
    ]
    assert len(candidate_entries) == 1
    assert candidate_entries[0]["raw_roi"] == -1.0
    assert candidate_entries[0]["shrunk_roi"] == -0.1
    assert candidate_entries[0]["risk_label"] == "insufficient_support"


def test_reliability_memory_detects_repeated_toxic_veto_and_scores_row() -> None:
    rows = [_candidate_disagreement_row(-4.0), _candidate_disagreement_row(-5.0), _candidate_disagreement_row(-3.0)]
    memory = build_reliability_memory(
        rows,
        prior_filled=0.0,
        min_filled_for_label=3,
        toxic_roi_threshold=-0.05,
    )
    toxic = [
        item
        for item in memory["top_toxic"]
        if item["expert"] == "candidate_agreement_expert" and item["event"] == "veto" and item["scope"] == "expert"
    ]
    assert toxic
    assert toxic[0]["risk_label"] == "toxic"

    score = score_row_reliability(_candidate_disagreement_row(-1.0), memory)
    assert score["has_toxic_veto"] is True
    assert score["toxic_veto_count"] >= 1
