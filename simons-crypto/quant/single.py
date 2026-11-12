"""One safe trade per week (adopted 2026-11-12 after research_single.py).

Rule: buy BTC spot with stablecoins only when BTC closes above its 50-day average AND there was no
shock (no BTC day beyond 3x its normal move in the last 7 days). Stop 2x ATR, target 3x ATR, sell
after 7 days. Otherwise hold stablecoins. Backtest (48 weeks): +10%, worst week -5.8%, max drawdown
-12%, versus -17% / -37% drawdown for holding BTC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import tradingview_link
from .strategy import Trade

RISK_PER_TRADE = 0.01   # lose at most 1% of the account if the stop is hit
MAX_POSITION = 0.50     # never more than half the account in BTC


def btc_signal(btc: pd.DataFrame) -> dict:
    c = btc["close"]
    r = np.log(c).diff()
    ma50 = float(c.rolling(50).mean().iloc[-1])
    sd = float(r.iloc[-37:-7].std())
    worst = float(r.tail(7).abs().max())
    prev = c.shift(1)
    tr = np.fmax(np.fmax(btc["high"] - btc["low"], (btc["high"] - prev).abs()), (btc["low"] - prev).abs())
    return {"price": float(c.iloc[-1]), "ma50": ma50, "uptrend": bool(c.iloc[-1] > ma50),
            "shock": bool(worst > 3 * sd), "shock_sigma": worst / sd if sd > 0 else float("nan"),
            "atr": float(tr.rolling(14).mean().iloc[-1])}


def weekly_trade(equity: float, btc: pd.DataFrame, mark: float | None = None,
                 risk_headlines: list[str] | None = None) -> tuple[Trade, dict]:
    s = btc_signal(btc)
    entry = float(mark or s["price"])
    reasons = []
    if not s["uptrend"]:
        reasons.append(f"BTC ${s['price']:,.0f} is below its 50-day average ${s['ma50']:,.0f} (downtrend)")
    if s["shock"]:
        reasons.append(f"Market shock: a BTC day moved {s['shock_sigma']:.1f}x its normal range in the last 7 days")
    if risk_headlines:
        reasons.append(f"Risk headlines dominate BTC news: {risk_headlines[0]}")
    if reasons:
        return Trade("THIS WEEK'S TRADE", "USDC", "HOLD STABLECOINS", "Your own wallet or a trusted exchange",
                     float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, 7, "n/a",
                     reasons + ["Conditions to buy BTC: close above the 50-day average and no shock for 7 days",
                                "Idle stablecoins: USDC lending on a major platform or Aave (~4-8%)"],
                     {"tradingview": tradingview_link("BTC")}, s), s
    stop, target = entry - 2 * s["atr"], entry + 3 * s["atr"]
    units = equity * RISK_PER_TRADE / (entry - stop)
    units = min(units, equity * MAX_POSITION / entry)
    return Trade("THIS WEEK'S TRADE", "BTC", "BUY (spot, paid with stablecoins)", "Any major spot exchange",
                 entry, stop, target, units, units * entry, units * (entry - stop), 7, "Rule-based (backtested)",
                 [f"BTC ${s['price']:,.0f} is above its 50-day average ${s['ma50']:,.0f}",
                  "No market shock in the last 7 days",
                  f"Stop 2x daily range (${2 * s['atr']:,.0f}), target 3x, sell after 7 days if neither is hit"],
                 {"tradingview": tradingview_link("BTC")}, s), s
