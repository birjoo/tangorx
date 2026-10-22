"""Where to put stablecoins: ranked, fee-adjusted, price-neutral opportunities.

1. Spot-perp carry      buy the coin with USDC, short its perp. Earns funding.
2. Cross-venue funding  short the perp where funding is highest, long it where lowest. Earns the gap.
3. HLP vault            deposit USDC into Hyperliquid's market-making vault.
Each is judged on a forecast (averages, not today's spike) minus trading costs.
"""
from __future__ import annotations

import pandas as pd

from . import data

HLP_VAULT = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
CARRY_FEE_APR = 0.104   # ~0.2% round trip on two legs, re-paid each week
XVENUE_FEE_APR = 0.156  # ~0.3%: two perp venues, taker both sides, re-paid each week


def spot_perp_carry(universe: pd.DataFrame, funding_hist: pd.DataFrame, min_volume: float = 20e6) -> pd.DataFrame:
    rows = []
    for coin, u in universe[universe["volume_24h_usd"] >= min_volume].iterrows():
        h = funding_hist[coin].dropna().tail(14) if coin in funding_hist else pd.Series(dtype=float)
        if len(h) < 10:
            continue
        avg = float(h.mean() * 24 * 365)
        rows.append({"coin": coin, "today_apr": u["funding_1h"] * 24 * 365, "avg14_apr": avg,
                     "positive_days": float((h > 0).mean()), "net_apr": avg - CARRY_FEE_APR,
                     "volume_24h_usd": u["volume_24h_usd"]})
    return pd.DataFrame(rows).set_index("coin").sort_values("net_apr", ascending=False) if rows else pd.DataFrame()


def cross_venue(universe: pd.DataFrame, funding_hist: pd.DataFrame, min_volume: float = 20e6) -> pd.DataFrame:
    """Funding gaps between Hyperliquid and Binance/Bybit. Only the Hyperliquid leg has history,
    so a gap counts only when Hyperliquid's 14-day average agrees with today's direction."""
    try:
        cv = data.cross_venue_funding()
    except Exception:  # noqa: BLE001 - optional board
        return pd.DataFrame()
    liquid = universe.index[universe["volume_24h_usd"] >= min_volume]
    cv = cv[cv.index.isin(liquid)].dropna(subset=["HlPerp"])
    rows = []
    for coin, r in cv.iterrows():
        others = r.drop("HlPerp").dropna()
        if others.empty:
            continue
        venue = (others - r["HlPerp"]).abs().idxmax()
        gap = r["HlPerp"] - others[venue]
        h = funding_hist[coin].dropna().tail(14) if coin in funding_hist else pd.Series(dtype=float)
        hl_avg = float(h.mean() * 24 * 365) if len(h) else float("nan")
        # Hyperliquid's average must sit on the same side of the other venue as today
        persistent = len(h) >= 10 and (hl_avg - others[venue]) * gap > 0
        rows.append({"coin": coin, "hl_apr": r["HlPerp"], "other_venue": venue.replace("Perp", ""),
                     "other_apr": others[venue], "gap_apr": gap, "hl_avg14_apr": hl_avg,
                     "persistent": persistent, "net_apr": abs(gap) - XVENUE_FEE_APR})
    out = pd.DataFrame(rows)
    return out.set_index("coin").sort_values("net_apr", ascending=False) if len(out) else out


def hlp_apr() -> float:
    try:
        return float(data._post(data.HL_INFO, {"type": "vaultDetails", "vaultAddress": HLP_VAULT})["apr"])
    except Exception:  # noqa: BLE001
        return float("nan")
