#!/usr/bin/env python3
"""Weekly Simons-style crypto picks.

  python run_weekly.py                 # live run, writes reports/YYYY-Www.md
  python run_weekly.py --demo          # offline run on synthetic data
  python run_weekly.py --equity 25000  # size trades for your account
  python run_weekly.py evaluate        # score past picks in reports/journal.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from quant import data
from quant.backtest import build_model
from quant.news import coin_news
from quant.report import append_journal, render
from quant.signals import atr, compute_signals
from quant.backtest import FIXED_V2
from quant.signals import zscore_xs
from quant.stablecoin import cross_venue, hlp_apr, spot_perp_carry
from quant import strategy
from quant.strategy import (Config, carry_candidates, pick_spec_futures, pick_spot_trades, pick_stablecoin,
                            regime)


def stable_boards(carry: pd.DataFrame, xv: pd.DataFrame, hlp: float) -> str:
    L = ["", "## Stablecoin boards (forecast = 14-day average, net of fees)", ""]
    if len(carry):
        L += ["**Spot-perp carry**", "", "| Coin | Today | 14d avg | Positive days | Net APR |", "|---|---|---|---|---|"]
        L += [f"| {c} | {r.today_apr:+.0%} | {r.avg14_apr:+.0%} | {r.positive_days:.0%} | {r.net_apr:+.0%} |"
              for c, r in carry.head(6).iterrows()] + [""]
    if len(xv):
        L += ["**Cross-exchange funding gaps**", "", "| Coin | Hyperliquid | Other | Gap | HL 14d avg | Persistent | Net APR |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {c} | {r.hl_apr:+.0%} | {r.other_venue} {r.other_apr:+.0%} | {r.gap_apr:+.0%} | "
              f"{r.hl_avg14_apr:+.0%} | {'yes' if r.persistent else 'no'} | {r.net_apr:+.0%} |"
              for c, r in xv.head(6).iterrows()] + [""]
    if hlp == hlp:
        L += [f"HLP vault (USDC): {hlp:.1%} APR", ""]
    return "\n".join(L)

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"


def gather_live(cfg: Config, n_coins: int, days: int):
    print("· Hyperliquid universe ...", file=sys.stderr)
    universe = data.perp_universe()
    coins = universe.sort_values("volume_24h_usd", ascending=False).head(n_coins).index.tolist()
    print(f"· candles + funding for {len(coins)} perps ({days}d) ...", file=sys.stderr)
    panel = data.load_panel(coins, days, cache_dir=str(ROOT / ".cache"))
    print("· TradingView, news, Fear&Greed, stablecoin pegs ...", file=sys.stderr)
    return (panel, universe, pd.DataFrame(), data.tradingview_ratings(coins), data.headlines(),
            data.fear_greed(), data.stablecoin_pegs())


def gather_demo():
    from quant.synthetic import make_dex, make_market
    panel, universe = make_market()
    heads = pd.DataFrame({"title": ["C03 protocol exploited for $40M", "Bitcoin ETF inflows rise"],
                          "published": pd.Timestamp.now(tz="UTC"), "source": "demo"})
    fng = pd.Series([55.0], index=[pd.Timestamp.now(tz="UTC")])
    return panel, universe, make_dex(), pd.Series(dtype=float), heads, fng, pd.DataFrame()


def run(args) -> Path:
    cfg = Config(equity=args.equity)
    panel, universe, dex, tv, heads, fng, pegs = gather_demo() if args.demo else gather_live(cfg, args.coins, args.days)

    sig = compute_signals(panel)
    model = build_model(sig, panel["close"], fixed=FIXED_V2)
    atr_df = atr(panel)
    reg = regime(panel["close"], fng)
    strategy.SHOCK[0] = reg["shock"]
    news_ctx = coin_news(panel["close"].columns.tolist(), heads)
    universe = universe.loc[universe.index.isin(panel["close"].columns)]
    session_z = (zscore_xs(sig["asia_minus_us"]).iloc[-1] if "asia_minus_us" in sig else pd.Series(dtype=float))

    spots = pick_spot_trades(cfg, model, universe, atr_df, reg, news_ctx, tv, n=2)
    carry = spot_perp_carry(universe, panel["funding"], cfg.min_futures_volume)
    xv = pd.DataFrame() if args.demo else cross_venue(universe, panel["funding"], cfg.min_futures_volume)
    hlp = float("nan") if args.demo else hlp_apr()
    stable = pick_stablecoin(cfg, reg, carry, xv, hlp)
    spec = pick_spec_futures(cfg, model, universe, atr_df, reg, news_ctx, tv, session_z,
                             exclude={t.symbol for t in spots} | {stable.symbol})
    trades = spots + [stable] + ([spec] if spec else [])

    asof = panel["close"].index[-1]
    REPORTS.mkdir(exist_ok=True)
    stem = f"{asof:%G-W%V}" + ("-demo" if args.demo else "")
    md = render(asof, trades, model, reg, cfg, pegs, carry_candidates(cfg, universe), heads, demo=args.demo)
    md += stable_boards(carry, xv, hlp)
    out = REPORTS / f"{stem}.md"
    out.write_text(md)
    (REPORTS / f"{stem}.json").write_text(json.dumps(
        {"asof": str(asof), "regime": reg, "model": model.summary, "trades": [t.to_dict() for t in trades]},
        indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)))
    if not args.demo:
        append_journal(REPORTS / "journal.csv", asof, trades)
    print(md)
    print(f"\nSaved {out}", file=sys.stderr)
    return out


def evaluate(_args) -> None:
    """Score closed picks: did price hit stop or target first within 7 days (daily bars)?"""
    path = REPORTS / "journal.csv"
    if not path.exists():
        sys.exit("No journal yet. Run a live week first.")
    j = pd.read_csv(path, parse_dates=["date"])
    rows = []
    for _, t in j.iterrows():
        coin = str(t["symbol"])
        if "(" in coin or pd.isna(t["stop"]) or (pd.Timestamp.now() - t["date"]).days < 8:
            continue  # DEX tokens / carry trades / open trades: score manually from fills
        bars = data.daily_candles(coin, (pd.Timestamp.now() - t["date"]).days + 2)
        bars = bars[bars.index.tz_convert(None) > t["date"]].head(7)
        if bars.empty:
            continue
        long = t["side"] == "LONG"
        r, outcome = (bars["close"].iloc[-1] / t["entry"] - 1) * (1 if long else -1), "time"
        for _, b in bars.iterrows():
            hit_stop = b["low"] <= t["stop"] if long else b["high"] >= t["stop"]
            hit_tgt = b["high"] >= t["target"] if long else b["low"] <= t["target"]
            if hit_stop:  # conservative: stop wins ties
                r, outcome = (t["stop"] / t["entry"] - 1) * (1 if long else -1), "stop"
                break
            if hit_tgt:
                r, outcome = (t["target"] / t["entry"] - 1) * (1 if long else -1), "target"
                break
        rows.append({"date": t["date"].date(), "symbol": coin, "side": t["side"], "outcome": outcome,
                     "return": r, "pnl_usd": r * t["notional_usd"]})
    if not rows:
        sys.exit("No closed, auto-scorable trades yet.")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTrades {len(df)} | hit rate {(df['return'] > 0).mean():.0%} | "
          f"avg return {df['return'].mean():+.2%} | total PnL ${df['pnl_usd'].sum():,.0f}")


def status(_args) -> None:
    """Open trades (last 8 days): live price, stop/target hits so far, unrealised P&L."""
    path = REPORTS / "journal.csv"
    if not path.exists():
        sys.exit("No journal yet.")
    j = pd.read_csv(path, parse_dates=["date"])
    j = j[(pd.Timestamp.now() - j["date"]).dt.days <= 8].dropna(subset=["stop"])
    marks = data.perp_universe()["mark"]
    rows = []
    for _, t in j.iterrows():
        coin, long = str(t["symbol"]), t["side"] == "LONG"
        bars = data.daily_candles(coin, 12)
        bars = bars[bars.index.tz_convert(None) > t["date"]]
        sgn = 1 if long else -1
        state = "open"
        for _, b in bars.iterrows():
            if (b["low"] <= t["stop"]) if long else (b["high"] >= t["stop"]):
                state = "STOPPED"
                break
            if (b["high"] >= t["target"]) if long else (b["low"] <= t["target"]):
                state = "TARGET HIT"
                break
        px = float(marks.get(coin, float("nan")))
        exit_px = t["stop"] if state == "STOPPED" else t["target"] if state == "TARGET HIT" else px
        r = (exit_px / t["entry"] - 1) * sgn
        rows.append({"opened": t["date"].date(), "trade": t["bucket"][:24], "coin": coin, "side": t["side"],
                     "entry": t["entry"], "now": px, "state": state, "return": f"{r:+.1%}",
                     "pnl_usd": round(r * t["notional_usd"], 2),
                     "days_left": max(0, 7 - (pd.Timestamp.now() - t["date"]).days)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nOpen/closed P&L: ${df['pnl_usd'].sum():,.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=["run", "evaluate", "status"])
    p.add_argument("--demo", action="store_true", help="offline synthetic data")
    p.add_argument("--equity", type=float, default=10_000)
    p.add_argument("--coins", type=int, default=60, help="perps in the model universe (by volume)")
    p.add_argument("--days", type=int, default=400, help="history length for validation")
    args = p.parse_args()
    {"evaluate": evaluate, "status": status}.get(args.command, run)(args)


if __name__ == "__main__":
    main()
