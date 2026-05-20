# BTC Polymarket Trading Bot — Research Study

![status](https://img.shields.io/badge/status-research%20study-orange)
![python](https://img.shields.io/badge/python-3.12-blue)
![tests](https://img.shields.io/badge/tests-226%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

**A production-grade research system that tests whether a high-accuracy
short-horizon BTC direction model can be traded profitably on Polymarket's
5-minute "Up/Down" markets — and documents, under a strict evidence protocol,
why it cannot.**

This repository is published as a **negative-result research study**. The
prediction model works: it reaches ~79% directional accuracy on 5-minute BTC
windows. The trading system built around it does not make money — and the
interesting part is *exactly why*. Everything here is reproducible so that
others can verify the result and reuse the infrastructure.

---

## TL;DR

- **Model:** rolling-logistic and consensus models predict the direction of BTC
  over a 5-minute window. Measured directional accuracy: **79.2%**
  (1,098 / 1,386 resolved windows — see `data/evidence/monitor_5m_state.json`).
- **Outcome:** that accuracy **does not convert into profit.** Across 775
  simulated fills the strategy produced **−$127.62** net PnL at a profit factor
  of 0.96 — structurally below break-even.
- **Root cause:** *adverse selection.* The model uses public Binance data; the
  Polymarket order book is priced off the **same** public data. When the model
  is confident, the market already is too, so the token costs $0.90+ and the
  asymmetric binary payoff (lose 100% of stake, win only the remainder) erases
  the edge.
- **Verdict (machine-readable, `data/evidence/phaseAF_forward_gate.md`):**
  `release_tier = research_only`, `paper_or_live_allowed = false`. The project's
  own release gate forbids deploying capital — by design.
- **Why publish it:** the result is real, the methodology is rigorous, and the
  infrastructure (data pipeline, evidence gating, walk-forward backtester) is
  reusable. A documented negative result with a hard gate is worth more than an
  undocumented bot that quietly loses money.

---

## The core finding

A high directional hit-rate is **not** an edge on a prediction market.

| What the model sees               | What the market sees  | Token price | Result                    |
|-----------------------------------|-----------------------|-------------|---------------------------|
| BTC clearly moving up             | the same Binance feed | $0.90+      | no margin left to capture |
| BTC ambiguous, "good" price $0.50 | the same ambiguity    | $0.50       | model is also near 50/50  |

The strategy systematically selects the trades where it has **no** informational
advantage, and the binary asymmetric payoff turns a 70% win rate into a
profit factor below 1.0. An audit of expected-value calibration
(`Phase AB`) found the ranking *inverted*: the top `expected_edge` quintile lost
**−$156**, the bottom quintile made **+$8**. More confidence correlated with
*worse* trades.

This is not an implementation bug. It is a structural property of trading a
public-data directional signal against an informed market. No threshold tuning
or filter on top of the signal repairs it — and this repository contains the
experiments (`Phase S` through `Phase AD`) that demonstrate each attempt failing.

## What's in here

A complete, modular trading-research stack — kept as a worked example of
non-trivial system architecture:

- **Data & market layer** (`src/polymarket/`, `src/data/`) — Polymarket Gamma /
  CLOB clients, market discovery, order-book depth capture (top-5/10, VWAP,
  microprice), dynamic fee-rate model, book-quality gates.
- **Models** (`src/strategy/`) — rolling online `LogisticRegression`
  (`spot_logistic`), window-path regression, a transferred Markov/transformer
  model, and a consensus blender.
- **Decision & risk** (`src/strategy/`, `src/runtime/`) — EV-aware decision
  logic, confirmation gate, adaptive risk layer, "X3" risk resolver, position
  sizing, timing policy.
- **Execution** (`src/execution/`, `src/polymarket/execution.py`) — paper-trade
  simulation with a realistic maker fill-probability model, shadow auditor.
- **Backtesting** (`src/backtest/`) — causal walk-forward engine (no lookahead),
  redecision replay, fill-execution audit, profit diagnostics.
- **Evidence protocol** (`src/`, `data/evidence/`) — every release candidate
  passes (or fails) a machine-readable forward-evidence gate before any capital
  is allowed. In this project, it always failed — correctly.

~16,600 lines of Python across `src/`, 226 passing tests.

## Results & evidence

All evidence ships in the repository so the result can be checked independently:

- `data/evidence/monitor_5m_state.json` — live monitor state: 1,386 resolved
  predictions, 79.2% accuracy.
- `data/evidence/phaseAF_forward_gate.md` / `.json` — the hard release gate
  verdict (`research_only`, paper & live forbidden).
- `data/evidence/phaseAF_monitor_metrics.md` — leakage-clean, no-retune,
  zero contamination rows.
- `data/sample/` — sample `predictions_5m.jsonl` / `outcomes_5m.jsonl` so the
  accuracy and replay numbers can be recomputed without a live run.

## Reproducing

```bash
git clone <this-repo>
cd btc-polymarket-trading-bot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# run the test suite (226 tests)
pytest -q

# run the accuracy monitor (observation only, no trading)
python3 -m src.main --config configs/default.yaml --mode monitor-5m

# causal walk-forward backtest (no lookahead)
python3 -m src.main --config configs/default.yaml --mode backtest-walkforward
```

No API keys are needed for monitoring, backtesting, or tests. Live wiring
requires a Polymarket wallet — see "Safety model" below.

## Repository layout

```
src/            trading-research stack (data, models, decision, execution, backtest)
configs/        runtime configs (default, monitor, paper example)
scripts/        backtest / monitor / replay / reporting helpers
tests/          226 unit & regression tests
data/evidence/  machine-readable release-gate verdicts and monitor metrics
data/sample/    sample prediction/outcome rows for reproduction
models/         model artifacts
```

## Safety model

This is research software. Live trading is intentionally hard to enable and is
**off** by default. A live order requires **all** of:

1. `execution.live_trading: true` in the config,
2. environment variable `LIVE_TRADING=true`,
3. the runtime flag `--confirm-live`.

No secrets are committed. All credentials are read from the environment —
copy `.env.example` to `.env` and fill it only if you intend live wiring.
Given the documented negative result, **running this live is not advised.**

## Contact

Questions, corrections, and reproduction notes are welcome:

**falcons.keenest0q@icloud.com**

## Disclaimer

This project is a research study, not financial advice and not a profitable
trading system. Its documented conclusion is that the strategy **loses money**.
It is published for its methodology, its architecture, and its honest negative
result. Do not trade real capital with it.

## License

MIT — see [LICENSE](LICENSE).
