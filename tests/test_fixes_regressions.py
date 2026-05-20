from datetime import UTC, datetime, timedelta

from src.backtest.simulator import BacktestStep, run_simple_backtest
from src.polymarket.execution import OrderIntent
from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.polymarket.risk import RiskManager
from src.runtime import timing as timing_mod
from src.runtime.modes import sim as sim_mod
from src.strategy.calibration import compute_adaptive_model_weight, compute_online_bias_shift
from src.strategy.model_wrapper import MarkovProxyModel, ModelContext
from src.strategy.signals import decide_trade


def test_backtest_down_token_payout_is_correct() -> None:
    intent = OrderIntent(
        market_slug="m",
        token_id="down_token",
        side="BUY",
        order_type="TAKER",
        price=0.4,
        size=1.0,
        expected_edge=0.1,
    )
    step = BacktestStep(
        intent=intent,
        resolved_side="down",
        up_token_id="up_token",
        down_token_id="down_token",
        spread=0.01,
    )
    metrics = run_simple_backtest([step], seed=42)
    assert metrics.trades == 1
    assert abs(metrics.net_pnl - 0.6) < 1e-9


def test_cooldown_expires_in_windows() -> None:
    risk = RiskManager(
        max_exposure_per_window_usd=100.0,
        max_daily_loss_usd=1000.0,
        cooldown_after_loss_streak=1,
        cooldown_windows=2,
        max_open_positions=3,
    )
    base = datetime.fromisoformat(risk.state.date_utc + "T12:00:00+00:00").astimezone(UTC)
    risk.on_trade_open()
    risk.on_trade_close(-1.0)

    assert risk.can_trade(base, exposure_usd=10.0)[0] is False
    assert risk.can_trade(base + timedelta(minutes=5), exposure_usd=10.0)[0] is False
    assert risk.can_trade(base + timedelta(minutes=10), exposure_usd=10.0)[0] is True


def test_maker_preference_picks_maker_when_close_to_taker_ev() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.50, 0.501, 0.5005, 0.001, 0.0),
        down=OutcomeBook("down", 0.499, 0.50, 0.4995, 0.001, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.503,
        min_edge_to_trade=0.0001,
        min_edge_for_taker=0.0001,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, maker_fee_bps=0.0, taker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        maker_fill_probability=0.95,
        maker_ev_advantage_required=0.005,
    )
    assert d.action == "trade"
    assert d.intent is not None
    assert d.intent.order_type == "MAKER"


def test_online_calibration_adjusts_bias_and_weight() -> None:
    rows = [
        {"actual_side": "up", "p_up": 0.30, "model_p_up": 0.30, "market_p_up": 0.70}
        for _ in range(30)
    ]
    shift = compute_online_bias_shift(rows, lookback=30, min_samples=5, strength=1.0, max_abs_shift=0.2)
    weight = compute_adaptive_model_weight(
        rows,
        lookback=30,
        min_samples=5,
        default_weight=0.5,
        min_weight=0.1,
        max_weight=0.9,
    )
    assert shift > 0
    assert weight < 0.5


def test_proxy_model_neutral_context_stays_neutral() -> None:
    model = MarkovProxyModel(default_confidence=0.5)
    p_up = model.predict_proba(
        ModelContext(
            up_mid=0.50,
            down_mid=0.50,
            up_imbalance=0.0,
            down_imbalance=0.0,
            seconds_to_expiry=240.0,
        )
    )
    assert abs(p_up - 0.5) < 1e-9


def test_run_sim_uses_execution_config_without_name_error(tmp_path, monkeypatch) -> None:
    class _Logger:
        def info(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(sim_mod, "build_logger", lambda *args, **kwargs: _Logger())
    monkeypatch.setattr(sim_mod, "GammaClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(sim_mod, "ClobClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(sim_mod, "discover_candidate_markets", lambda *args, **kwargs: [])

    cfg = {
        "app": {"name": "btc_polymarket_bot"},
        "paths": {"logs_dir": str(tmp_path), "outputs_dir": str(tmp_path)},
        "polymarket": {"gamma_base_url": "https://gamma", "clob_base_url": "https://clob", "lookahead_windows": 1},
        "risk": {
            "max_exposure_per_window_usd": 100.0,
            "max_daily_loss_usd": 1000.0,
            "cooldown_after_loss_streak": 3,
            "cooldown_windows": 3,
            "max_open_positions": 1,
            "drift_enabled": False,
        },
        "execution": {
            "live_trading": False,
            "require_confirm_live": True,
            "allowed_order_types": ["maker"],
        },
        "fees": {"maker_fee_bps": 0.0, "taker_fee_bps": 0.0, "curve_rate": 0.0, "curve_exponent": 1.0, "min_fee": 0.0},
        "model": {"default_confidence": 0.5},
        "book_quality": {},
    }

    summary = sim_mod.run_sim(cfg, confirm_live=False)
    assert summary["mode"] == "SIM"
    assert summary["trades"] == []


def test_reduced_risk_window_multiplier_applies_for_warsaw_night() -> None:
    cfg = {
        "reduced_risk_timezone": "Europe/Warsaw",
        "reduced_risk_start_local": "00:00",
        "reduced_risk_end_local": "05:30",
        "reduced_risk_exposure_multiplier": 0.5,
    }
    night_utc = datetime(2026, 3, 16, 2, 15, tzinfo=UTC)
    day_utc = datetime(2026, 3, 16, 12, 15, tzinfo=UTC)
    assert timing_mod.risk_window_exposure_multiplier(now=night_utc, risk_cfg=cfg) == 0.5
    assert timing_mod.risk_window_exposure_multiplier(now=day_utc, risk_cfg=cfg) == 1.0
