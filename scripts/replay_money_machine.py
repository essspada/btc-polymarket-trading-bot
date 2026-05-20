#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.money_machine import (  # noqa: E402
    DEFAULT_BANKROLLS,
    DEFAULT_SIZINGS,
    DEFAULT_STRATEGIES,
    build_replay_report,
    empirical_p_up_series,
    load_resolved_rows,
)


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "0.0%"


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Money-machine replay")
    lines.append("")
    assumptions = payload.get("assumptions") or {}
    warning = str(assumptions.get("warning") or "").strip()
    if warning:
        lines.append("> Warning: legacy research replay, not runtime-parity forward evidence.")
        lines.append(">")
        lines.append(f"> {warning}")
        lines.append("")
    lines.append(f"Rows: `{payload.get('rows')}`")
    lines.append(f"Window: `{payload.get('first_created_at')}` - `{payload.get('last_created_at')}`")
    lines.append("")
    lines.append("## Fixed $100 Per Trade")
    lines.append("")
    lines.append("| strategy | trades | win rate | PnL | worst 12h | adverse 10pp | adverse 15pp |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(payload.get("fixed_cash_100") or [], key=lambda item: float(item.get("pnl", 0.0)), reverse=True):
        lines.append(
            "| {strategy} | {trades} | {wr} | {pnl} | {w12} | {a10} | {a15} |".format(
                strategy=row.get("strategy"),
                trades=row.get("trades"),
                wr=_fmt_pct(float(row.get("win_rate", 0.0)) * 100.0),
                pnl=_fmt_money(row.get("pnl")),
                w12=_fmt_money(row.get("worst_12h")),
                a10=_fmt_money(row.get("adverse_10pp")),
                a15=_fmt_money(row.get("adverse_15pp")),
            )
        )
    lines.append("")
    lines.append("## Bankroll Ranking")
    lines.append("")
    lines.append("| rank | strategy | sizing | min return | avg return | max DD | min trades |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for idx, row in enumerate((payload.get("ranking") or [])[:20], start=1):
        lines.append(
            "| {idx} | {strategy} | {sizing} | {min_ret} | {avg_ret} | {dd} | {trades} |".format(
                idx=idx,
                strategy=row.get("strategy"),
                sizing=row.get("sizing"),
                min_ret=_fmt_pct(row.get("min_return_pct")),
                avg_ret=_fmt_pct(row.get("avg_return_pct")),
                dd=_fmt_pct(row.get("max_drawdown_pct")),
                trades=row.get("min_trades"),
            )
        )
    lines.append("")
    rec = payload.get("recommended") or {}
    lines.append("## Recommended Candidate Replay")
    lines.append("")
    lines.append(f"Strategy: `{rec.get('strategy')}`")
    lines.append(f"Sizing: `{rec.get('sizing')}`")
    lines.append("")
    lines.append("| initial | final | PnL | return | max DD | trades | win rate | profit factor |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rec.get("bankroll") or []:
        lines.append(
            "| {initial} | {final} | {pnl} | {ret} | {dd} | {trades} | {wr} | {pf:.2f} |".format(
                initial=_fmt_money(row.get("initial_balance")),
                final=_fmt_money(row.get("final_balance")),
                pnl=_fmt_money(row.get("pnl")),
                ret=_fmt_pct(row.get("return_pct")),
                dd=_fmt_pct(row.get("max_drawdown_pct")),
                trades=row.get("trades"),
                wr=_fmt_pct(float(row.get("win_rate", 0.0)) * 100.0),
                pf=float(row.get("profit_factor", 0.0)),
            )
        )
    lines.append("")
    lines.append("## Adverse Fill Stress")
    lines.append("")
    lines.append("Stress model: winning maker fills are reduced by the gap; losing maker fills still execute. This is conservative and is not a real fill model.")
    lines.append("")
    lines.append("| gap | initial | final | PnL | return | max DD |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in rec.get("adverse_fill_stress") or []:
        gap = 1.0 - float(row.get("win_fill_factor", 1.0))
        lines.append(
            "| {gap} | {initial} | {final} | {pnl} | {ret} | {dd} |".format(
                gap=f"{gap * 100:.0f}pp",
                initial=_fmt_money(row.get("initial_balance")),
                final=_fmt_money(row.get("final_balance")),
                pnl=_fmt_money(row.get("pnl")),
                ret=_fmt_pct(row.get("return_pct")),
                dd=_fmt_pct(row.get("max_drawdown_pct")),
            )
        )
    lines.append("")
    lines.append("## Diagnostics For Recommended")
    lines.append("")
    lines.append("### worst 12h segments")
    lines.append("")
    lines.append("| segment UTC | trades | win rate | PnL |")
    lines.append("|---|---:|---:|---:|")
    for row in sorted(rec.get("segments_12h") or [], key=lambda item: float(item.get("pnl", 0.0)))[:8]:
        lines.append(
            "| {segment} | {trades} | {wr} | {pnl} |".format(
                segment=row.get("segment_start_utc"),
                trades=row.get("trades"),
                wr=_fmt_pct(float(row.get("win_rate", 0.0)) * 100.0),
                pnl=_fmt_money(row.get("pnl")),
            )
        )
    lines.append("")
    diagnostics = (rec.get("diagnostics") or {})
    for name in ("confidence", "edge", "price", "side"):
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| bucket | trades | win rate | PnL |")
        lines.append("|---|---:|---:|---:|")
        for row in diagnostics.get(name) or []:
            lines.append(
                "| {bucket} | {trades} | {wr} | {pnl} |".format(
                    bucket=row.get("bucket"),
                    trades=row.get("trades"),
                    wr=_fmt_pct(float(row.get("win_rate", 0.0)) * 100.0),
                    pnl=_fmt_money(row.get("pnl")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay BTC Polymarket money-machine candidates on resolved monitor rows.")
    parser.add_argument(
        "--rows",
        default="data/sample/outcomes_5m.jsonl",
    )
    parser.add_argument("--out-json", default="reports/money_machine_replay.json")
    parser.add_argument("--out-md", default="reports/money_machine_replay.md")
    parser.add_argument("--recommended-strategy", default="proxy_confirmed_pup_edge003")
    parser.add_argument("--recommended-sizing", default="flat_8pct_cap150")
    args = parser.parse_args()

    rows = load_resolved_rows(Path(args.rows))
    empirical = empirical_p_up_series(rows, warmup_rows=120, nearest_k=120)
    payload = build_replay_report(
        rows,
        empirical_probs=empirical,
        strategies=DEFAULT_STRATEGIES,
        sizings=DEFAULT_SIZINGS,
        bankrolls=DEFAULT_BANKROLLS,
        recommended_strategy=str(args.recommended_strategy),
        recommended_sizing=str(args.recommended_sizing),
    )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
