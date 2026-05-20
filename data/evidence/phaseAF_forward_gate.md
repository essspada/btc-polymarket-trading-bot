# Phase AF Forward Evidence Gate

Generated: `2026-05-05T17:11:15.396092+00:00`

## Verdict

- status: `collect_more`
- classification: `research_only_collect_more`
- monitor_only_allowed: `true`
- paper_smoke_allowed: `false`
- paper_trial_allowed: `false`
- paper_safety_pass: `false`
- business_target_pass: `false`
- stress_pass: `false`
- live_allowed: `false`
- next_action: collect no-retune unseen rows; no paper/live promotion

## Blockers

| severity | tier | code | actual | required | detail |
|---|---|---|---:|---:|---|
| `collect_more` | `paper_smoke` | `days_after_freeze_below_min` | `6.621834799861111` | `7` | days_after_freeze below required minimum |
| `collect_more` | `paper_smoke` | `resolved_prediction_rows_below_min` | `1378.0` | `1500` | resolved_prediction_rows below required minimum |
| `collect_more` | `stress` | `missing_remove_best_trade_pnl` | `` | `> 0.0` | remove_best_trade_pnl is unavailable |
| `collect_more` | `stress` | `missing_fee_slippage_10pct_pnl` | `` | `> 0.0` | fee_slippage_10pct_pnl is unavailable |
| `collect_more` | `stress` | `missing_loser_first_max_drawdown_pct` | `` | `<= 0.25` | loser_first_max_drawdown_pct is unavailable |
| `collect_more` | `stress` | `missing_without_top_positive_quintile_pnl` | `` | `0.0` | without_top_positive_quintile_pnl is unavailable |

## Safety

- `+50%` ROI is a business target, not a safety shortcut.
- Live remains forbidden by this gate even for strong paper candidates.
- Any retune after seeing holdout requires a new freeze and resets evidence.
