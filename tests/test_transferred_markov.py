from src.strategy.transferred_markov import TransferredMarkovPredictor


def test_map_probs_to_up_basic_directional() -> None:
    p = TransferredMarkovPredictor.map_probs_to_up(buy_prob=0.30, sell_prob=0.50, hold_prob=0.20, hold_weight=0.0)
    assert abs(p - (0.30 / 0.80)) < 1e-9


def test_map_probs_to_up_with_hold_weight() -> None:
    p = TransferredMarkovPredictor.map_probs_to_up(buy_prob=0.40, sell_prob=0.40, hold_prob=0.20, hold_weight=1.0)
    assert abs(p - 0.5) < 1e-9


def test_map_probs_to_up_handles_zero_denominator() -> None:
    p = TransferredMarkovPredictor.map_probs_to_up(buy_prob=0.0, sell_prob=0.0, hold_prob=0.0, hold_weight=0.0)
    assert p == 0.5
