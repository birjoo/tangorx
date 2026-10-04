"""Synthetic market with a planted edge, for demos and tests (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(n_coins: int = 30, days: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    coins = ["BTC", "ETH", "SOL"] + [f"C{i:02d}" for i in range(n_coins - 3)]
    idx = pd.date_range(end=pd.Timestamp("2026-10-04", tz="UTC"), periods=days, freq="D")
    market = rng.standard_t(4, days) * 0.025
    vol = rng.uniform(0.02, 0.06, n_coins)
    rets = np.zeros((days, n_coins))
    fund = rng.normal(0.00001, 0.00002, (days, n_coins))
    for t in range(1, days):
        # planted edges: 20d momentum continues weakly; high funding underperforms
        mom = rets[max(0, t - 20):t].sum(axis=0)
        edge = 0.004 * np.sign(mom) - 30 * fund[t - 1]
        rets[t] = market[t] + edge + rng.standard_t(4, n_coins) * vol
    fund[:, 5] = 0.00005  # one coin with persistent high funding (~44% APR)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=coins)
    spread = np.abs(rng.normal(0, 1, (days, n_coins))) * vol
    high, low = close * (1 + spread), close * (1 - spread)
    dvol = pd.DataFrame(rng.lognormal(17, 1, (days, n_coins)), index=idx, columns=coins)
    panel = {"close": close, "high": high, "low": low, "dollar_volume": dvol,
             "funding": pd.DataFrame(fund, index=idx, columns=coins)}
    universe = pd.DataFrame({
        "max_leverage": 20, "mark": close.iloc[-1], "funding_1h": panel["funding"].iloc[-1],
        "open_interest_usd": dvol.iloc[-1] * 2, "volume_24h_usd": dvol.iloc[-1] * 3,
        "prev_day_px": close.iloc[-2]})
    return panel, universe


def make_dex(n: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    liq = rng.lognormal(12.5, 1.2, n)
    return pd.DataFrame({
        "symbol": [f"MEME{i}" for i in range(n)], "name": "x", "chain": "solana", "dex": "raydium",
        "address": [f"addr{i}" for i in range(n)], "url": [f"https://dexscreener.com/solana/addr{i}" for i in range(n)],
        "price_usd": rng.lognormal(-6, 2, n), "liquidity_usd": liq, "fdv": liq * rng.uniform(3, 80, n),
        "market_cap": liq * 10, "vol_h24": liq * rng.uniform(0.5, 30, n), "vol_h6": liq * rng.uniform(0.1, 8, n),
        "vol_h1": liq * 0.1, "chg_h1": rng.normal(0, 5, n), "chg_h6": rng.normal(0, 15, n),
        "chg_h24": rng.normal(10, 40, n), "buys_h24": rng.integers(100, 5000, n),
        "sells_h24": rng.integers(100, 5000, n), "age_days": rng.uniform(0, 120, n),
        "boost": rng.choice([0, 0, 100, 500], n)})
