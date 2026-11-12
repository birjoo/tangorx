"""Backtest: one spot long per week, safest version.

Each week (non-overlapping), among the 15 most-traded coins (trailing 30-day dollar volume, no look-ahead):
  buy at the close, stop at 2x ATR, target at 3x ATR, otherwise sell at the 7th close.
Rules compared:
  A  always buy the model's top pick
  B  A, only when BTC is above its 50-day average (else hold stablecoins: 0%)
  C  B, and no shock (no BTC day beyond 3x its normal move in the last 7 days)
  R  C's timing, but an equal-weight basket of all 15 liquid coins (does the pick add anything?)
  BTC buy-and-hold weekly, and BTC with C's timing
"""
import glob
import pickle

import numpy as np
import pandas as pd

from quant.backtest import FIXED_V2
from quant.signals import atr, compute_signals, zscore_xs

panel = pickle.load(open(sorted(glob.glob(".cache/panel-*.pkl"))[-1], "rb"))
close, high, low = panel["close"], panel["high"], panel["low"]
sig = compute_signals(panel)
z = {k: zscore_xs(sig[k]) for k in FIXED_V2}
comp = sum(z[k].fillna(0) * w for k, w in FIXED_V2.items())
A = atr(panel)
liq30 = panel["dollar_volume"].rolling(30, min_periods=20).mean()
btc = close["BTC"]
btc_r = np.log(btc).diff()
trend = btc > btc.rolling(50).mean()
sd = btc_r.rolling(30).std().shift(7)
shock = btc_r.abs().rolling(7).max() > 3 * sd

idx = close.index
dates = [d for d in idx[60::7] if idx.get_loc(d) + 7 < len(idx)]


def trade(coin, d):
    i = idx.get_loc(d)
    e, a = close.at[d, coin], A.at[d, coin]
    if not (e > 0 and a > 0):
        return np.nan
    stop, tgt = e - 2 * a, e + 3 * a
    for j in range(i + 1, i + 8):
        dd = idx[j]
        if low.at[dd, coin] <= stop:
            return stop / e - 1
        if high.at[dd, coin] >= tgt:
            return tgt / e - 1
    return close.iat[i + 7, close.columns.get_loc(coin)] / e - 1


rows = []
for d in dates:
    liquid = liq30.loc[d].dropna().nlargest(15).index
    c = comp.loc[d, liquid].dropna()
    c = c[close.loc[d, c.index].notna()]
    if c.empty:
        continue
    pick = c.idxmax()
    basket = np.nanmean([trade(x, d) for x in c.index])
    safe_b, safe_c = bool(trend.loc[d]), bool(trend.loc[d] and not shock.loc[d])
    r_pick, r_btc = trade(pick, d), trade("BTC", d)
    btc_hold = btc.iat[idx.get_loc(d) + 7] / btc.loc[d] - 1
    rows.append({"date": d, "pick": pick, "A": r_pick, "B": r_pick if safe_b else 0.0,
                 "C": r_pick if safe_c else 0.0, "R": basket if safe_c else 0.0,
                 "BTC hold": btc_hold, "BTC timed": r_btc if safe_c else 0.0, "in_market": safe_c})
df = pd.DataFrame(rows).set_index("date")


def stats(s):
    traded = s[s != 0]
    curve = (1 + s).cumprod()
    return {"weeks traded": f"{len(traded)}/{len(s)}", "total": f"{curve.iloc[-1] - 1:+.0%}",
            "avg/trade": f"{traded.mean():+.2%}" if len(traded) else "-",
            "win rate": f"{(traded > 0).mean():.0%}" if len(traded) else "-",
            "worst week": f"{s.min():+.1%}", "max drawdown": f"{(curve / curve.cummax() - 1).min():+.0%}",
            "sharpe": f"{s.mean() / s.std() * np.sqrt(52):+.2f}" if s.std() > 0 else "-"}


print(pd.DataFrame({k: stats(df[k]) for k in ["A", "B", "C", "R", "BTC hold", "BTC timed"]}).T.to_string())
h = len(df) // 2
print("\nRule C by half:", {lab: stats(x["C"])["total"] for lab, x in [("1st", df.iloc[:h]), ("2nd", df.iloc[h:])]})
print("Recent 10 weeks (C):")
print(df[["pick", "C", "in_market", "BTC hold"]].tail(10).round(3).to_string())
