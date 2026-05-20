"""Typed pydantic schema for runtime YAML configuration.

The trading runtime still consumes a plain ``dict[str, Any]`` in many places.
This module makes pydantic the authoritative config boundary: YAML is parsed,
validated, defaulted, and only then converted to the legacy dict shape. That
keeps the migration low-risk while making bad configs fail at process startup.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OrderType = Literal["maker", "taker"]


def _default_maker_order_types() -> list[OrderType]:
    return ["maker"]


class ConfigSection(BaseModel):
    """Base for nested sections.

    ``extra='allow'`` is intentional for nested sections: the project has a few
    research-only knobs and scripts that add runtime metadata. Top-level keys are
    still forbidden by ``AppConfig`` so misspelled sections fail fast.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class AppSection(ConfigSection):
    name: str = "btc_polymarket_bot"
    mode: str = "monitor-5m"
    seed: int = 42


class PolymarketSection(ConfigSection):
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    ws_market_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    search_query: str = "Bitcoin Up or Down"
    target_series_slug: str = "btc-up-or-down-5m"
    fallback_series_slugs: list[str] = Field(default_factory=lambda: ["btc-up-or-down-15m", "btc-up-or-down-hourly"])
    lookahead_windows: int = Field(default=12, ge=0)


class DiscoverySection(ConfigSection):
    require_accepting_orders: bool = True
    include_closed: bool = False
    require_chainlink_for_5m: bool = True
    max_results: int = Field(default=50, ge=1)


class MonitorSection(ConfigSection):
    poll_seconds: float = Field(default=10.0, gt=0)
    lookback_windows: int = Field(default=1, ge=0)
    lookahead_windows: int = Field(default=12, ge=0)
    min_start_delay_seconds: float = Field(default=120.0, ge=0)
    max_start_delay_seconds: float = Field(default=210.0, ge=0)
    settlement_grace_seconds: float = Field(default=45.0, ge=0)
    resolution_winner_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    max_runtime_minutes: float = Field(default=0.0, ge=0)
    predictions_file: str = "predictions_5m.jsonl"
    outcomes_file: str = "outcomes_5m.jsonl"
    state_file: str = "monitor_5m_state.json"
    timing_events_file: str = "monitor_timing_events_5m.jsonl"
    adaptive_risk_shadow_file: str = "monitor_adaptive_risk_shadow_decisions_5m.jsonl"

    @model_validator(mode="after")
    def validate_start_delay_window(self) -> MonitorSection:
        if self.max_start_delay_seconds < self.min_start_delay_seconds:
            raise ValueError("monitor.max_start_delay_seconds must be >= min_start_delay_seconds")
        return self


class ConfidenceSkipBandSection(ConfigSection):
    enabled: bool = False
    low: float = Field(default=0.65, ge=0.0, le=1.0)
    high: float = Field(default=0.85, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_band(self) -> ConfidenceSkipBandSection:
        if self.high <= self.low:
            raise ValueError("execution.confidence_skip_band.high must be > low")
        return self


class ActionabilityCalibrationSection(ConfigSection):
    enabled: bool = False
    artifact_path: str | None = None
    min_bucket_samples: int = Field(default=20, ge=0)
    fallback_policy: str = "conservative_shrink"
    conservative_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    min_calibrated_net_edge: float | None = None
    min_breakeven_margin: float | None = None


class ConfirmationGateSection(ConfigSection):
    enabled: bool = False
    candidate_key: str = "proxy_logistic_market_blend"
    min_edge: float = 0.03
    require_same_side: bool = True
    max_spread: float = Field(default=0.02, ge=0.0)
    min_price: float = Field(default=0.05, ge=0.0, le=1.0)
    max_price: float = Field(default=0.95, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_price_window(self) -> ConfirmationGateSection:
        if self.max_price < self.min_price:
            raise ValueError("execution.confirmation_gate.max_price must be >= min_price")
        return self


class ExecutionSection(ConfigSection):
    live_trading: bool = False
    require_confirm_live: bool = True
    live_order_state_file: str = "live_order_state.json"
    maker_preference: bool = True
    lock_side_to_prediction: bool = True
    allowed_order_types: list[OrderType] = Field(default_factory=_default_maker_order_types)
    min_edge_to_trade: float = 0.004
    min_edge_for_taker: float = 0.009
    max_entry_price: float | None = None
    sizing_mode: str = "edge_scaled"
    min_reward_to_risk_ratio: float = 0.0
    min_expected_roi_cash: float | None = None
    min_breakeven_margin: float | None = None
    confidence_skip_band: ConfidenceSkipBandSection = Field(default_factory=ConfidenceSkipBandSection)
    actionability_calibration: ActionabilityCalibrationSection = Field(default_factory=ActionabilityCalibrationSection)
    maker_fill_probability: float = Field(default=0.65, ge=0.0, le=1.0)
    maker_fill_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    maker_fill_cap: float = Field(default=0.9, ge=0.0, le=1.0)
    maker_fill_base: float = Field(default=0.78, ge=0.0, le=1.0)
    maker_fill_spread_penalty: float = Field(default=7.0, ge=0.0)
    maker_fill_late_penalty_90: float = Field(default=0.15, ge=0.0)
    maker_fill_late_penalty_45: float = Field(default=0.10, ge=0.0)
    maker_ev_advantage_required: float = 0.0005
    confirmation_gate: ConfirmationGateSection = Field(default_factory=ConfirmationGateSection)
    slippage_bps_taker: float = Field(default=12.0, ge=0.0)
    slippage_bps_maker: float = Field(default=3.0, ge=0.0)
    latency_buffer_seconds: float = Field(default=20.0, ge=0.0)

    @field_validator("allowed_order_types")
    @classmethod
    def validate_allowed_order_types(cls, value: list[OrderType]) -> list[OrderType]:
        if not value:
            raise ValueError("execution.allowed_order_types must contain at least one order type")
        return value

    @model_validator(mode="after")
    def validate_execution(self) -> ExecutionSection:
        if self.maker_fill_cap < self.maker_fill_floor:
            raise ValueError("execution.maker_fill_cap must be >= maker_fill_floor")
        if self.max_entry_price is not None and not 0.0 <= self.max_entry_price <= 1.0:
            raise ValueError("execution.max_entry_price must be between 0 and 1")
        return self


class FeesSection(ConfigSection):
    maker_fee_bps: float = Field(default=0.0, ge=0.0)
    taker_fee_bps: float = Field(default=1000.0, ge=0.0)
    maker_fee_rate: float = Field(default=0.0, ge=0.0)
    taker_fee_rate: float = Field(default=0.072, ge=0.0)
    curve_rate: float = Field(default=0.25, ge=0.0)
    curve_exponent: float = Field(default=2.0, ge=0.0)
    min_fee: float = Field(default=0.0001, ge=0.0)


class SizingCapSection(ConfigSection):
    enabled: bool = False
    max_trade_usd: float | None = None
    min_trade_usd_after_cap: float | None = None
    max_fraction_of_base_exposure: float | None = None


class X3ResolverSection(ConfigSection):
    enabled: bool = False
    mode: str = "shadow"
    policy: str = "cap_large_cash12"
    large_cash_threshold_usd: float = Field(default=16.0, ge=0.0)
    large_cash_cap_usd: float = Field(default=12.0, ge=0.0)
    mid_price_min: float = Field(default=0.55, ge=0.0, le=1.0)
    mid_price_max: float = Field(default=0.75, ge=0.0, le=1.0)
    mid_price_cap_usd: float | None = None
    up_cap_usd: float | None = None
    down_cap_usd: float | None = None
    global_cap_usd: float | None = None
    hard_max_cash_usd: float | None = None
    min_trade_usd_after_cap: float = Field(default=5.0, ge=0.0)
    respect_advisor_cap: bool = False
    respect_advisor_veto: bool = False

    @model_validator(mode="after")
    def validate_mid_price_window(self) -> X3ResolverSection:
        if self.mid_price_max < self.mid_price_min:
            raise ValueError("risk.x3_resolver.mid_price_max must be >= mid_price_min")
        return self


class AdaptiveRiskSection(ConfigSection):
    enabled: bool = True
    mode: str = "shadow"
    drawdown_soft_pct: float = Field(default=0.05, ge=0.0)
    drawdown_hard_pct: float = Field(default=0.15, ge=0.0)
    daily_loss_soft_frac: float = Field(default=0.06, ge=0.0)
    daily_loss_lock_frac: float = Field(default=0.12, ge=0.0)
    loss_streak_cooldown: int = Field(default=2, ge=0)
    loss_streak_long_cooldown: int = Field(default=3, ge=0)
    cooldown_windows: int = Field(default=1, ge=0)
    long_cooldown_windows: int = Field(default=3, ge=0)
    min_multiplier: float = Field(default=0.35, ge=0.0, le=1.0)
    veto_negative_economics: bool = True
    min_expected_roi_cash: float = 0.0
    min_breakeven_margin: float = 0.0
    max_candidate_fraction_of_equity: float = Field(default=0.12, ge=0.0)
    high_volatility_bps: float = Field(default=20.0, ge=0.0)
    extreme_volatility_bps: float = Field(default=35.0, ge=0.0)
    high_volatility_multiplier: float = Field(default=0.65, ge=0.0, le=1.0)
    wide_spread_norm: float = Field(default=0.06, ge=0.0)
    extreme_spread_norm: float = Field(default=0.10, ge=0.0)
    wide_spread_multiplier: float = Field(default=0.65, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(default=0.58, ge=0.0, le=1.0)
    low_confidence_multiplier: float = Field(default=0.75, ge=0.0, le=1.0)
    min_trade_usd_after_cap: float = Field(default=0.0, ge=0.0)
    monitor_shadow_equity_usd: float = Field(default=100.0, gt=0.0)


class StatefulRegimeSection(ConfigSection):
    enabled: bool = False
    drawdown_soft_pct: float = Field(default=0.08, ge=0.0)
    drawdown_hard_pct: float = Field(default=0.18, ge=0.0)
    loss_streak_soft: int = Field(default=2, ge=0)
    loss_streak_hard: int = Field(default=4, ge=0)
    daily_loss_soft_frac: float = Field(default=0.08, ge=0.0)
    daily_loss_hard_frac: float = Field(default=0.18, ge=0.0)
    min_multiplier: float = Field(default=0.35, ge=0.0, le=1.0)
    recovery_wins_required: int = Field(default=2, ge=0)


class RiskSection(ConfigSection):
    max_exposure_per_window_usd: float = Field(default=150.0, ge=0.0)
    max_balance_fraction_per_trade: float = Field(default=0.12, ge=0.0)
    sizing_cap: SizingCapSection = Field(default_factory=SizingCapSection)
    x3_resolver: X3ResolverSection = Field(default_factory=X3ResolverSection)
    adaptive_risk: AdaptiveRiskSection = Field(default_factory=AdaptiveRiskSection)
    stateful_regime: StatefulRegimeSection = Field(default_factory=StatefulRegimeSection)
    reduced_risk_timezone: str = "Europe/Warsaw"
    reduced_risk_start_local: str = "00:00"
    reduced_risk_end_local: str = "05:30"
    reduced_risk_exposure_multiplier: float = Field(default=0.6, ge=0.0, le=1.0)
    max_daily_loss_usd: float = Field(default=300.0, ge=0.0)
    cooldown_after_loss_streak: int = Field(default=3, ge=0)
    cooldown_windows: int = Field(default=3, ge=0)
    max_open_positions: int = Field(default=3, ge=0)
    drift_enabled: bool = True
    drift_lookback: int = Field(default=60, ge=1)
    drift_min_samples: int = Field(default=40, ge=1)
    drift_min_accuracy: float = Field(default=0.46, ge=0.0, le=1.0)
    drift_cooldown_windows: int = Field(default=6, ge=0)


class BookQualitySection(ConfigSection):
    sentinel_bid: float = Field(default=0.01, ge=0.0, le=1.0)
    sentinel_ask: float = Field(default=0.99, ge=0.0, le=1.0)
    max_spread: float = Field(default=0.08, ge=0.0)
    max_midpoint_sum_deviation: float = Field(default=0.05, ge=0.0)
    min_midpoint: float = Field(default=0.02, ge=0.0, le=1.0)
    max_midpoint: float = Field(default=0.98, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_midpoint_window(self) -> BookQualitySection:
        if self.max_midpoint < self.min_midpoint:
            raise ValueError("book_quality.max_midpoint must be >= min_midpoint")
        return self


class OracleSection(ConfigSection):
    enabled: bool = False
    symbol: str = "BTCUSDT"
    oracle_price_url: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0.0)


class SpotContextSection(ConfigSection):
    enabled: bool = True
    symbol: str = "BTCUSDT"
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    kline_limit: int = Field(default=8, ge=1)
    cache_ttl_seconds: float = Field(default=5.0, ge=0.0)


class SpotWindowPathSection(ConfigSection):
    min_sigma_bps: float = Field(default=4.0, ge=0.0)
    drift_weight: float = 0.25
    prob_shrink: float = Field(default=0.85, ge=0.0, le=1.0)
    spot_model_weight: float = Field(default=0.45, ge=0.0, le=1.0)


class SpotLogisticSection(ConfigSection):
    lookback: int = Field(default=240, ge=1)
    min_train_samples: int = Field(default=40, ge=1)
    min_class_samples: int = Field(default=10, ge=1)
    seed_rows_file: str = "outputs/archive_2026-03-11/native_archive_dataset_spot.jsonl"


class SpotLogisticMarketBlendSection(ConfigSection):
    logistic_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class SpotConsensusBlendSection(ConfigSection):
    path_weight: float = Field(default=0.5, ge=0.0)
    logistic_weight: float = Field(default=0.5, ge=0.0)


class ProxyLogisticMarketBlendSection(ConfigSection):
    proxy_weight: float = Field(default=0.1, ge=0.0)
    logistic_weight: float = Field(default=0.2, ge=0.0)


class ProxyLogisticMetaPolicySection(ConfigSection):
    proxy_midpoint_band: float = Field(default=0.06, ge=0.0)
    proxy_edge_margin: float = 0.0
    fallback_source: str = "spot_logistic_online"


class TimingPolicyRuntimeSection(ConfigSection):
    allow_late_fresh_start: bool = False
    late_start_grace_seconds: float = Field(default=10.0, ge=0.0)


class TimingPolicyStep(ConfigSection):
    delay_seconds: float = Field(default=0.0, ge=0.0)
    candidate: str
    extra_edge: float = 0.0


class AdaptiveBlendSection(ConfigSection):
    lookback_resolved: int = Field(default=120, ge=1)
    min_samples: int = Field(default=20, ge=1)
    default_model_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    min_model_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    max_model_weight: float = Field(default=0.9, ge=0.0, le=1.0)
    calibration_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    max_calibration_shift: float = Field(default=0.15, ge=0.0)
    platt_lookback: int = Field(default=240, ge=1)
    platt_min_samples: int = Field(default=40, ge=1)
    platt_min_class_samples: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_weight_bounds(self) -> AdaptiveBlendSection:
        if self.max_model_weight < self.min_model_weight:
            raise ValueError("model.adaptive_blend.max_model_weight must be >= min_model_weight")
        return self


class TransferredMarkovSection(ConfigSection):
    enabled: bool = False
    python_bin: str = ""
    project_root: str = ""
    predictor_module: str = "src.models.predict_sequence"
    model_config_path: str = ""
    model_weights_path: str = ""
    output_dir: str = "outputs/markov_bridge"
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    candles_limit: int = Field(default=14000, ge=1)
    request_limit: int = Field(default=1000, ge=1)
    request_timeout_seconds: float = Field(default=20.0, gt=0.0)
    request_pause_seconds: float = Field(default=0.1, ge=0.0)
    predict_timeout_seconds: float = Field(default=120.0, gt=0.0)
    hold_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_prediction_age_minutes: float = Field(default=45.0, gt=0.0)
    min_refresh_seconds: float = Field(default=30.0, ge=0.0)


class ModelSection(ConfigSection):
    source: str = "log180_then_log240"
    fallback_to_proxy_orderbook: bool = True
    name: str = "markov_proxy_port"
    source_model_path: str = "models/eurusd36m_hard_20260213_211208_hybrid.pt"
    source_model_note: str = ""
    calibration: str = "platt"
    default_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    spot_window_path: SpotWindowPathSection = Field(default_factory=SpotWindowPathSection)
    spot_logistic: SpotLogisticSection = Field(default_factory=SpotLogisticSection)
    spot_logistic_market_blend: SpotLogisticMarketBlendSection = Field(default_factory=SpotLogisticMarketBlendSection)
    spot_consensus_blend: SpotConsensusBlendSection = Field(default_factory=SpotConsensusBlendSection)
    proxy_logistic_market_blend: ProxyLogisticMarketBlendSection = Field(default_factory=ProxyLogisticMarketBlendSection)
    proxy_logistic_meta_policy: ProxyLogisticMetaPolicySection = Field(default_factory=ProxyLogisticMetaPolicySection)
    timing_policy_runtime: TimingPolicyRuntimeSection = Field(default_factory=TimingPolicyRuntimeSection)
    timing_policy_base_sources: dict[str, str] = Field(
        default_factory=lambda: {
            "cons180_then_log240": "spot_consensus_blend",
            "log240_only": "spot_consensus_blend",
            "log180_then_log240": "spot_consensus_blend",
            "path180_then_path240": "spot_consensus_blend",
        }
    )
    timing_policies: dict[str, list[TimingPolicyStep]] = Field(
        default_factory=lambda: {
            "log180_then_log240": [
                TimingPolicyStep(delay_seconds=180, candidate="spot_logistic_online", extra_edge=0.10),
                TimingPolicyStep(delay_seconds=240, candidate="spot_logistic_online", extra_edge=0.0),
            ]
        }
    )
    adaptive_blend: AdaptiveBlendSection = Field(default_factory=AdaptiveBlendSection)
    transferred_markov: TransferredMarkovSection = Field(default_factory=TransferredMarkovSection)


class PathsSection(ConfigSection):
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"
    reports_dir: str = "reports"


class PaperSection(ConfigSection):
    initial_balance_usd: float = Field(default=100.0, gt=0.0)
    max_runtime_minutes: float = Field(default=0.0, ge=0.0)
    predictions_file: str = "paper_predictions_5m.jsonl"
    outcomes_file: str = "paper_outcomes_5m.jsonl"
    trades_file: str = "paper_trades_5m.jsonl"
    x3_shadow_file: str = "paper_x3_shadow_decisions_5m.jsonl"
    adaptive_risk_shadow_file: str = "paper_adaptive_risk_shadow_decisions_5m.jsonl"
    state_file: str = "paper_5m_state.json"
    timing_events_file: str = "paper_timing_events_5m.jsonl"


class LiveLikeBacktestSection(ConfigSection):
    use_normalized_entry_prices: bool = True
    max_raw_entry_sum_deviation: float = Field(default=0.02, ge=0.0)


class WalkforwardBacktestSection(ConfigSection):
    source_outcomes_file: str = "paper_outcomes_5m.jsonl"
    lookback: int = Field(default=240, ge=1)
    min_train_samples: int = Field(default=40, ge=1)
    allowed_order_types: list[OrderType] = Field(default_factory=_default_maker_order_types)
    require_book_quality: bool = True
    max_outcome_spread: float = Field(default=0.25, ge=0.0)
    min_edge_to_trade: float = 0.004
    min_edge_for_taker: float = 0.009


class BacktestSection(ConfigSection):
    live_like: LiveLikeBacktestSection = Field(default_factory=LiveLikeBacktestSection)
    walkforward: WalkforwardBacktestSection = Field(default_factory=WalkforwardBacktestSection)


class AppConfig(BaseModel):
    """Root application config.

    The root forbids unknown top-level sections because those are almost always
    typos. Nested sections allow extra research knobs for backward compatibility.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    app: AppSection = Field(default_factory=AppSection)
    polymarket: PolymarketSection = Field(default_factory=PolymarketSection)
    discovery: DiscoverySection = Field(default_factory=DiscoverySection)
    monitor: MonitorSection = Field(default_factory=MonitorSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    fees: FeesSection = Field(default_factory=FeesSection)
    risk: RiskSection = Field(default_factory=RiskSection)
    book_quality: BookQualitySection = Field(default_factory=BookQualitySection)
    oracle: OracleSection = Field(default_factory=OracleSection)
    spot_context: SpotContextSection = Field(default_factory=SpotContextSection)
    model: ModelSection = Field(default_factory=ModelSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    paper: PaperSection = Field(default_factory=PaperSection)
    backtest: BacktestSection = Field(default_factory=BacktestSection)

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return the legacy dict shape consumed by the current runtime."""
        return self.model_dump(mode="python")
