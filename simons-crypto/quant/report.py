"""Markdown weekly report + trade journal."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

from .strategy import Trade


def _px(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:.6g}"


def render(asof: pd.Timestamp, trades: list[Trade], model, reg: dict, cfg, pegs: pd.DataFrame,
           carry: pd.DataFrame, heads: pd.DataFrame, demo: bool = False) -> str:
    s = model.summary
    L = [f"# Weekly Quant Picks — {asof:%Y-%m-%d} (week {asof:%G-W%V})", ""]
    if demo:
        L += ["> **DEMO RUN on synthetic data. Not real prices. Do not trade this.**", ""]
    L += ["> Systematic, statistical, small edges, strict risk. Not financial advice. Paper-trade first.", "",
          "## Regime", "",
          f"- BTC above 50-day average: **{'yes' if reg['btc_above_50d'] else 'no'}**",
          f"- BTC 30d realised vol (ann.): {reg['btc_vol_ann']:.0%}" if reg['btc_vol_ann'] == reg['btc_vol_ann'] else "- BTC vol: n/a",
          f"- Fear & Greed: {reg['fear_greed']:.0f}" if reg['fear_greed'] == reg['fear_greed'] else "- Fear & Greed: n/a",
          f"- Risk multiplier applied to all sizes: **{reg['risk_multiplier']:.2f}x**", "",
          "## Model health (walk-forward, out-of-sample)", "",
          f"- Weeks tested: {s['oos_weeks']}",
          f"- Mean rank IC: {s['oos_mean_ic']:+.3f}  (0.02–0.05 is a real edge at scale)",
          f"- IC hit rate: {s['oos_ic_hit_rate']:.0%}",
          f"- Top-minus-bottom quintile Sharpe (ann.): {s['oos_ls_sharpe']:.2f}",
          f"- Verdict: **{'TRADEABLE' if model.trustworthy else 'NOT VALIDATED — trade at half size or paper only'}**", ""]
    if not model.trustworthy:
        for t in trades:
            if t.bucket.startswith(("1", "2. FUTURES (directional")):
                t.rationale.insert(0, "Model failed OOS checks this week: halve size or paper-trade")

    L += [f"## This week's 3 trades (equity ${cfg.equity:,.0f})", ""]
    for t in trades:
        L += [f"### {t.bucket}: {t.symbol} — {t.side}", "",
              "| Field | Value |", "|---|---|",
              f"| Venue | {t.venue} |",
              f"| Entry | {_px(t.entry)} |",
              f"| Stop | {_px(t.stop)} |",
              f"| Target | {_px(t.target)} |",
              f"| Size | {t.size_units:,.4g} units (${t.notional_usd:,.0f} notional) |",
              f"| Max loss at stop | {'n/a (hedged; risk = funding flip / venue)' if math.isnan(t.stop) else f'${t.risk_usd:,.0f}'} |",
              f"| Horizon | {t.horizon_days} days (exit at market if neither stop nor target hit) |",
              f"| Conviction | {t.conviction} |", ""]
        L += [f"- {r}" for r in t.rationale]
        L += [f"- [{k}]({v})" for k, v in t.links.items()] + [""]
    if len(trades) < 3:
        L += [f"_Only {len(trades)} trade(s) passed filters. Unused risk stays in stablecoins — "
              "not trading is a position._", ""]

    L += ["## Signal weights (learned from data, not opinion)", "",
          "| Signal | Weeks | Mean IC | t-stat | Weight |", "|---|---|---|---|---|"]
    for name, r in model.weights.sort_values("weight", key=abs, ascending=False).iterrows():
        L.append(f"| {name} | {int(r['n_weeks'])} | {r['mean_ic']:+.3f} | {r['t_stat']:+.2f} | {r['weight']:+.2f} |")
    L += ["", "## Top / bottom of the composite ranking", "",
          "Longs: " + ", ".join(f"{c} ({v:+.2f})" for c, v in model.composite_today.head(8).items()),
          "",
          "Shorts: " + ", ".join(f"{c} ({v:+.2f})" for c, v in model.composite_today.tail(8)[::-1].items()), ""]

    if carry is not None and len(carry):
        L += ["## Funding carry board (stablecoin yield via delta-neutral)", "",
              "| Perp | Funding APR | 24h volume |", "|---|---|---|"]
        for c, r in carry.head(6).iterrows():
            L.append(f"| {c} | {r['funding_apr']:+.0%} | ${r['volume_24h_usd'] / 1e6:,.0f}M |")
        L.append("")
    if pegs is not None and len(pegs):
        L += ["## Stablecoin peg monitor", "", "| Stable | DEX price | Deviation |", "|---|---|---|"]
        for _, r in pegs.iterrows():
            flag = " ⚠️" if abs(r["deviation_bps"]) >= 50 else ""
            L.append(f"| {r['stable']} | {r['price']:.4f} | {r['deviation_bps']:+.0f} bps{flag} |")
        L.append("")
    if heads is not None and len(heads):
        L += ["## Headlines scanned (kill-switch input only)", ""]
        L += [f"- {h}" for h in heads["title"].head(10)] + [""]
    L += ["## Rules", "",
          "1. Enter within 24h of the report, limit orders only.",
          "2. Stops are placed immediately. Never moved further away.",
          "3. Exit everything after 7 days; next report re-decides from scratch.",
          "4. Log fills in `reports/journal.csv`; run `python run_weekly.py evaluate` to score the system.",
          "5. If total drawdown hits 15%, stop and re-validate the model.", ""]
    return "\n".join(L)


JOURNAL_FIELDS = ["date", "bucket", "symbol", "side", "entry", "stop", "target", "notional_usd", "risk_usd"]


def append_journal(path: Path, asof: pd.Timestamp, trades: list[Trade]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if new:
            w.writeheader()
        for t in trades:
            w.writerow({"date": f"{asof:%Y-%m-%d}", "bucket": t.bucket, "symbol": t.symbol, "side": t.side,
                        "entry": t.entry, "stop": t.stop, "target": t.target,
                        "notional_usd": round(t.notional_usd, 2), "risk_usd": round(t.risk_usd, 2)})
