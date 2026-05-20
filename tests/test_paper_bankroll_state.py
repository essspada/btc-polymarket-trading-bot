import json

from src.polymarket.fees import FeeModelConfig
from src.runtime.state import (
    PendingPaperTrade,
    count_active_pending_paper_trades,
    load_paper_bankroll_state,
)
from src.strategy.sizing import cap_exposure_by_balance


def _pending_trade(**overrides) -> PendingPaperTrade:
    base = {
        "trade_id": "t1",
        "market_id": "m1",
        "market_slug": "m1",
        "event_slug": "e1",
        "series_slug": "s1",
        "start_time": "2026-03-15T00:00:00+00:00",
        "end_time": "2026-03-15T00:05:00+00:00",
        "created_at": "2026-03-15T00:01:00+00:00",
        "predicted_side": "up",
        "p_up": 0.7,
        "p_down": 0.3,
        "model_source": "log180_then_log240",
        "token_id": "up1",
        "order_type": "TAKER",
        "fill_price": 0.55,
        "fill_size": 10.0,
        "filled": True,
        "fill_probability": 1.0,
        "expected_edge": 0.02,
        "up_token_id": "up1",
        "down_token_id": "down1",
        "up_best_ask": 0.55,
        "down_best_ask": 0.45,
        "cash_required": 12.0,
    }
    base.update(overrides)
    return PendingPaperTrade(**base)


def test_cap_exposure_by_balance_respects_fraction_and_balance() -> None:
    assert cap_exposure_by_balance(max_exposure_usd=150.0, balance_usd=100.0, max_balance_fraction_per_trade=0.12) == 12.0
    assert cap_exposure_by_balance(max_exposure_usd=8.0, balance_usd=100.0, max_balance_fraction_per_trade=0.12) == 8.0


def test_load_paper_bankroll_state_reconstructs_from_outputs(tmp_path) -> None:
    outcomes_path = tmp_path / "paper_outcomes_5m.jsonl"
    outcomes_path.write_text(
        json.dumps(
            {
                "market_id": "m0",
                "trade_id": "closed1",
                "trade_net_pnl": 5.0,
                "trade_fee": 1.1,
                "trade_slippage": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = load_paper_bankroll_state(
        state_path=tmp_path / "paper_5m_state.json",
        outcomes_path=outcomes_path,
        pending_trades={"m1": _pending_trade()},
        initial_balance=100.0,
        fee_cfg=FeeModelConfig(),
        exec_cfg={"slippage_bps_taker": 12.0, "slippage_bps_maker": 3.0},
    )
    assert state.initial_balance == 100.0
    assert state.available_cash == 93.0
    assert state.reserved_cash == 12.0
    assert state.realized_pnl == 5.0
    assert state.realized_fees == 1.1
    assert state.realized_slippage == 0.2
    assert state.settled_trades == 1
    assert state.opened_trades == 2


def test_load_paper_bankroll_state_prefers_persisted_state(tmp_path) -> None:
    state_path = tmp_path / "paper_5m_state.json"
    state_path.write_text(
        json.dumps(
            {
                "paper_bankroll": {
                    "initial_balance": 100.0,
                    "available_cash": 41.0,
                    "reserved_cash": 9.0,
                    "realized_pnl": 3.0,
                    "realized_fees": 0.7,
                    "realized_slippage": 0.1,
                    "opened_trades": 4,
                    "settled_trades": 3,
                }
            }
        ),
        encoding="utf-8",
    )
    state = load_paper_bankroll_state(
        state_path=state_path,
        outcomes_path=tmp_path / "paper_outcomes_5m.jsonl",
        pending_trades={"m1": _pending_trade()},
        initial_balance=100.0,
        fee_cfg=FeeModelConfig(),
        exec_cfg={"slippage_bps_taker": 12.0, "slippage_bps_maker": 3.0},
    )
    assert state.available_cash == 41.0
    assert state.reserved_cash == 9.0
    assert state.opened_trades == 4
    assert state.settled_trades == 3


def test_load_paper_bankroll_state_ignores_unfilled_pending_for_cash(tmp_path) -> None:
    outcomes_path = tmp_path / "paper_outcomes_5m.jsonl"
    outcomes_path.write_text(
        json.dumps(
            {
                "market_id": "m0",
                "trade_id": "closed1",
                "trade_net_pnl": 5.0,
                "trade_fee": 0.0,
                "trade_slippage": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = load_paper_bankroll_state(
        state_path=tmp_path / "paper_5m_state.json",
        outcomes_path=outcomes_path,
        pending_trades={
            "m1": _pending_trade(
                order_type="MAKER",
                filled=False,
                fill_size=0.0,
                fill_probability=0.49,
                cash_required=12.0,
            )
        },
        initial_balance=100.0,
        fee_cfg=FeeModelConfig(),
        exec_cfg={"slippage_bps_taker": 12.0, "slippage_bps_maker": 3.0},
    )
    assert state.available_cash == 105.0
    assert state.reserved_cash == 0.0
    assert state.opened_trades == 2
    assert state.settled_trades == 1


def test_count_active_pending_paper_trades_only_counts_filled() -> None:
    pending = {
        "m1": _pending_trade(filled=True, fill_size=10.0),
        "m2": _pending_trade(
            trade_id="t2",
            market_id="m2",
            market_slug="m2",
            order_type="MAKER",
            filled=False,
            fill_size=0.0,
            fill_probability=0.49,
            cash_required=9.0,
        ),
    }
    assert count_active_pending_paper_trades(pending) == 1
