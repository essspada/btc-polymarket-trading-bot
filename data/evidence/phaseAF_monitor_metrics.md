# Phase AF Monitor-Only Metrics

Run dir: `data/sample (350-row excerpt of this 1386-row monitor run)`

## Counts

- days_after_freeze: `6.621834799861111`
- prediction_rows: `1378`
- resolved_prediction_rows: `1378`
- shadow_trade_candidates: `572`
- volatility_regime_count: `3`
- leakage_clean: `True`
- no_retune: `True`
- contamination_rows: `0`

## Note

These metrics are monitor-only. They can satisfy the frozen shadow/smoke evidence tier, but cannot satisfy paper trial/pass because there are no filled paper trades.
