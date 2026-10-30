"""Turns model scores into exactly three sized, risk-defined trades per week.

  1. CORE SPOT      liquid large-cap, long only, validated composite model
  2. FUTURES        either a market-neutral funding-carry trade (stablecoin collateral)
                    or a directional perp trade (long or short) from the composite model
  3. SPECULATIVE    DEX token from DexScreener, hard rug filters, tiny size, unvalidated
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import tradingview_link
from .news import blocked


@dataclass
class Config:
    equity: float = 10_000.0
    risk_core: float = 0.010        # fraction of equity lost if stop is hit
    risk_futures: float = 0.010
    risk_spec: float = 0.0025
    max_spot_alloc: float = 0.25
    max_leverage: float = 3.0
    core_universe: int = 25         # top-N perps by volume treated as "large cap"
    min_futures_volume: float = 20e6
    carry_min_net_apr: float = 0.15 # forecast funding minus fees must clear this
    carry_fee_apr: float = 0.104    # ~0.2% round trip on both legs, re-paid weekly
    stop_atr: float = 2.0
    target_atr: float = 3.0
    # DEX rug filters
    dex_min_liquidity: float = 150_000
    dex_min_volume: float = 250_000
    dex_min_age_days: float = 7
    dex_max_fdv_to_liq: float = 60
    dex_max_vol_to_liq: float = 25  # above this smells like wash trading
    dex_stop: float = 0.25
    dex_target: float = 0.60


@dataclass
class Trade:
    bucket: str
    symbol: str
    side: str
    venue: str
    entry: float
    stop: float
    target: float
    size_units: float
    notional_usd: float
    risk_usd: float
    horizon_days: int
    conviction: str
    rationale: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def regime(close: pd.DataFrame, fng: pd.Series) -> dict:
    btc = close["BTC"].dropna() if "BTC" in close else pd.Series(dtype=float)
    up = bool(len(btc) >= 50 and btc.iloc[-1] > btc.rolling(50).mean().iloc[-1])
    vol = float(np.log(btc).diff().tail(30).std() * np.sqrt(365)) if len(btc) > 30 else float("nan")
    fg = float(fng.iloc[-1]) if len(fng) else float("nan")
    mult = 1.0 if up else 0.5
    if fg == fg and fg >= 80:
        mult *= 0.75  # euphoria: shrink gross
    # Shock rule (pre-set, not fitted): a BTC day beyond 3x its normal daily move in the last 7 days
    # means the statistical relationships the model relies on are suspended -> directional = paper only.
    r = np.log(btc).diff() if len(btc) > 40 else pd.Series(dtype=float)
    sd = float(r.iloc[-37:-7].std()) if len(r) else float("nan")
    worst = float(r.tail(7).abs().max()) if len(r) else float("nan")
    shock = bool(sd == sd and worst > 3 * sd)
    return {"btc_above_50d": up, "btc_vol_ann": vol, "fear_greed": fg, "risk_multiplier": mult,
            "shock": shock, "shock_sigma": worst / sd if sd == sd and sd > 0 else float("nan")}


def _size(cfg: Config, risk_frac: float, entry: float, stop: float, mult: float, cap_notional: float):
    risk_usd = cfg.equity * risk_frac * mult
    per_unit = abs(entry - stop)
    units = risk_usd / per_unit if per_unit > 0 else 0.0
    if units * entry > cap_notional:
        units = cap_notional / entry
    return units, units * entry, units * per_unit


SHOCK = [False]  # set by run_weekly from regime(); read by _directional
NEW_MODEL_CAP = 0.5  # v2 model adopted 2026-10-22: half size until it has 8 live weeks


def edge_multiplier(model) -> float:
    """Scale directional risk by the *recent* (last 26 weeks) out-of-sample Sharpe: full size only at 1.0+."""
    sh = model.summary.get("recent_ls_sharpe", model.summary["oos_ls_sharpe"])
    if not model.trustworthy or sh != sh or sh <= 0:
        return 0.25
    return float(min(NEW_MODEL_CAP, max(0.25, sh)))


def _conviction(score: float, trustworthy: bool, shock: bool = False) -> str:
    if shock:
        return "PAPER ONLY (market shock: model paused 7 days)"
    if not trustworthy:
        return "LOW (model not validated out-of-sample)"
    return "HIGH" if abs(score) >= 1.0 else "MEDIUM" if abs(score) >= 0.5 else "LOW"


def _directional(cfg, coin, side, score, model, atr_now, mark, mult, bucket, venue, cap, news_ctx, tv):
    sgn = 1 if side == "LONG" else -1
    a = float(atr_now)
    entry = float(mark)
    stop, target = entry - sgn * cfg.stop_atr * a, entry + sgn * cfg.target_atr * a
    units, notional, risk = _size(cfg, cfg.risk_core if bucket.startswith("1") else cfg.risk_futures,
                                  entry, stop, mult, cap)
    top = (model.weights["weight"].abs().sort_values(ascending=False).head(3).index.tolist())
    why = [f"Composite z-score {score:+.2f} (rank {list(model.composite_today.index).index(coin) + 1}"
           f"/{len(model.composite_today)})",
           f"Dominant validated signals: {', '.join(top)}",
           f"Stop {cfg.stop_atr}xATR, target {cfg.target_atr}xATR, 7-day time stop"]
    if coin in tv.index:
        why.append(f"TradingView technical rating {tv[coin]:+.2f} (context only)")
    if coin in news_ctx.index:
        why.append(f"{int(news_ctx.loc[coin, 'mentions'])} headlines this week (context only)")
    return Trade(bucket, coin, side, venue, entry, stop, target, units, notional, risk, 7,
                 _conviction(score, model.trustworthy, SHOCK[0]), why, {"tradingview": tradingview_link(coin)},
                 {"atr": a})


def pick_core_spot(cfg, model, universe, panel, atr_df, reg, news_ctx, tv) -> Trade | None:
    liquid = universe.sort_values("volume_24h_usd", ascending=False).head(cfg.core_universe).index
    for coin, score in model.composite_today.items():
        if coin not in liquid or score <= 0 or blocked(coin, news_ctx):
            continue
        return _directional(cfg, coin, "LONG", score, model, atr_df[coin].iloc[-1], universe.loc[coin, "mark"],
                            reg["risk_multiplier"] * edge_multiplier(model), "1. CORE SPOT", "Any major spot exchange (paid in USDC/USDT)",
                            cfg.equity * cfg.max_spot_alloc, news_ctx, tv)
    return None


def carry_candidates(cfg, universe) -> pd.DataFrame:
    u = universe[universe["volume_24h_usd"] >= cfg.min_futures_volume].copy()
    u["funding_apr"] = u["funding_1h"] * 24 * 365
    return u.sort_values("funding_apr", ascending=False)


def pick_futures(cfg, model, universe, panel, atr_df, reg, news_ctx, tv, exclude: set[str]) -> Trade | None:
    carry = carry_candidates(cfg, universe)
    fund_hist = panel["funding"]
    best = None
    for coin, row in carry.iterrows():
        if row["funding_apr"] < 0.05 or coin in exclude or blocked(coin, news_ctx):
            continue
        hist = fund_hist[coin].dropna().tail(14) if coin in fund_hist else pd.Series(dtype=float)
        if len(hist) < 10 or (hist > 0).mean() < 0.8:
            continue
        # Funding mean-reverts fast: forecast next week with the 14-day average, not today's print
        forecast = float(hist.mean() * 24 * 365)
        net = forecast - cfg.carry_fee_apr
        if net >= cfg.carry_min_net_apr and (best is None or net > best[2]):
            best = (coin, row, net, forecast, hist)
    if best:
        coin, row, net, forecast, hist = best
        notional = cfg.equity * 0.30 * reg["risk_multiplier"]  # per leg
        mark = float(row["mark"])
        return Trade("2. FUTURES (market-neutral carry)", coin, "LONG SPOT + SHORT PERP (1x, delta-neutral)",
                     "Spot on any CEX + short perp on Hyperliquid, USDC collateral", mark, float("nan"),
                     float("nan"), notional / mark, notional, 0.0, 7, "MEDIUM",
                     [f"14-day average funding {forecast:.0%} APR (today {row['funding_apr']:.0%}), positive on "
                      f"{(hist > 0).mean():.0%} of last {len(hist)} days",
                      f"After ~{cfg.carry_fee_apr:.0%} APR of trading fees: ~{net:.0%} APR, "
                      f"~${net / 52 * notional:,.0f}/week on ${notional:,.0f} per leg",
                      "Price-neutral: profit is the funding longs pay you; exit if APR < 10% or flips negative",
                      "Risks: funding flip, exchange risk, liquidation of perp leg if under-margined (keep >=50% margin)"],
                     {"tradingview": tradingview_link(coin)}, {"funding_apr": forecast, "net_apr": net})

    liquid = universe[universe["volume_24h_usd"] >= cfg.min_futures_volume].index
    comp = model.composite_today[model.composite_today.index.isin(liquid)]
    comp = comp[~comp.index.isin(exclude)]
    for coin in comp.abs().sort_values(ascending=False).index:
        if blocked(coin, news_ctx):
            continue
        score = float(comp[coin])
        side = "LONG" if score > 0 else "SHORT"
        cap = cfg.equity * cfg.max_leverage * 0.5
        t = _directional(cfg, coin, side, score, model, atr_df[coin].iloc[-1], universe.loc[coin, "mark"],
                         reg["risk_multiplier"] * edge_multiplier(model), "2. FUTURES (directional perp)",
                         f"Perpetual futures (e.g. Hyperliquid), isolated margin, <= {cfg.max_leverage:.0f}x",
                         cap, news_ctx, tv)
        t.extra["margin_at_max_lev"] = t.notional_usd / cfg.max_leverage
        t.extra["funding_apr"] = float(universe.loc[coin, "funding_1h"] * 24 * 365)
        return t
    return None


def score_dex(cfg, dex: pd.DataFrame) -> pd.DataFrame:
    if dex.empty:
        return dex
    d = dex.copy()
    d = d[~d["symbol"].str.upper().isin({"USDC", "USDT", "DAI", "WETH", "WBTC", "SOL", "WSOL", "ETH"})]
    d["buy_ratio"] = d["buys_h24"] / (d["buys_h24"] + d["sells_h24"]).replace(0, np.nan)
    d["fdv_to_liq"] = d["fdv"] / d["liquidity_usd"].replace(0, np.nan)
    d["vol_to_liq"] = d["vol_h24"] / d["liquidity_usd"].replace(0, np.nan)
    d["vol_accel"] = (d["vol_h6"] * 4) / d["vol_h24"].replace(0, np.nan)
    ok = ((d["liquidity_usd"] >= cfg.dex_min_liquidity) & (d["vol_h24"] >= cfg.dex_min_volume)
          & (d["age_days"] >= cfg.dex_min_age_days)
          & ((d["fdv_to_liq"] <= cfg.dex_max_fdv_to_liq) | (d["liquidity_usd"] >= 2_000_000))
          & (d["vol_to_liq"] <= cfg.dex_max_vol_to_liq) & (d["buy_ratio"] >= 0.5)
          & (d["chg_h24"] > -40) & (d["chg_h24"] < 300) & (d["buys_h24"] + d["sells_h24"] >= 300))
    d = d[ok].copy()
    if d.empty:
        return d

    def z(s):
        s = s.astype(float)
        return ((s - s.mean()) / s.std()).fillna(0).clip(-3, 3) if s.std() > 0 else s * 0

    d["score"] = (0.30 * z(np.sign(d["chg_h24"]) * np.log1p(d["chg_h24"].abs()))
                  + 0.15 * z(d["chg_h6"].clip(-50, 100))
                  + 0.25 * z(d["buy_ratio"])
                  + 0.20 * z(d["vol_accel"].clip(0, 4))
                  + 0.20 * z(np.log(d["liquidity_usd"]))
                  - 0.25 * z(np.log1p(d["boost"])))  # paid promotion is a contrarian signal
    return d.sort_values("score", ascending=False)


def pick_speculative(cfg, dex_scored: pd.DataFrame, reg) -> Trade | None:
    if dex_scored is None or len(dex_scored) < 5:
        return None  # ranking 1-4 tokens against each other is noise, not a signal
    r = dex_scored.iloc[0]
    entry = float(r["price_usd"])
    stop, target = entry * (1 - cfg.dex_stop), entry * (1 + cfg.dex_target)
    units, notional, risk = _size(cfg, cfg.risk_spec, entry, stop, reg["risk_multiplier"], cfg.equity * 0.03)
    return Trade("3. SPECULATIVE (DEX)", f"{r['symbol']} ({r['chain']})", "LONG", f"{r['dex']} on {r['chain']}",
                 entry, stop, target, units, notional, risk, 7,
                 "LOW (unvalidated, lottery-ticket sizing)",
                 [f"Liquidity ${r['liquidity_usd']:,.0f}, 24h vol ${r['vol_h24']:,.0f}, age {r['age_days']:.0f}d",
                  f"24h {r['chg_h24']:+.0f}%, 6h {r['chg_h6']:+.0f}%, buy ratio {r['buy_ratio']:.0%}",
                  f"FDV/liquidity {r['fdv_to_liq']:.1f}x, vol/liquidity {r['vol_to_liq']:.1f}x",
                  f"Passed rug filters; ranked #1 of {len(dex_scored)} survivors",
                  "Verify contract on a token scanner (honeypot/mint authority) before buying",
                  "Use limit orders, max 1% slippage; size so a total loss is irrelevant"],
                 {"dexscreener": r["url"]}, {"address": r["address"], "score": float(r["score"])})


# ---------------------------------------------------------------- v2 slots (2026-10-22)
def pick_spot_trades(cfg, model, universe, atr_df, reg, news_ctx, tv, n: int = 2) -> list[Trade]:
    """Top-n longs among the most liquid coins, bought with stablecoins. Core risk is split across them."""
    liquid = universe.sort_values("volume_24h_usd", ascending=False).head(cfg.core_universe).index
    out = []
    for coin, score in model.composite_today.items():
        if len(out) == n:
            break
        if coin not in liquid or score <= 0 or blocked(coin, news_ctx):
            continue
        out.append(_directional(cfg, coin, "LONG", score, model, atr_df[coin].iloc[-1], universe.loc[coin, "mark"],
                                reg["risk_multiplier"] * edge_multiplier(model) / n,
                                f"1{'ab'[len(out)]}. SPOT (paid with stablecoins)",
                                "Any major spot exchange, buy with USDC/USDT", cfg.equity * cfg.max_spot_alloc / n,
                                news_ctx, tv))
    return out


def pick_spec_futures(cfg, model, universe, atr_df, reg, news_ctx, tv, session_z: pd.Series,
                      exclude: set[str]) -> Trade | None:
    """'Showing promise': a mid-cap (not a top-10 major) with a positive composite and the strongest
    Asia-session buying. Long perp, half the normal futures risk."""
    liquid = universe[universe["volume_24h_usd"] >= cfg.min_futures_volume]
    majors = set(universe.sort_values("volume_24h_usd", ascending=False).head(10).index)
    cands = [c for c, s in model.composite_today.items()
             if s > 0 and c in liquid.index and c not in majors and c not in exclude
             and session_z.get(c, float("nan")) > 0 and not blocked(c, news_ctx)]
    if not cands:
        return None
    coin = max(cands, key=lambda c: session_z[c])
    t = _directional(cfg, coin, "LONG", float(model.composite_today[coin]), model, atr_df[coin].iloc[-1],
                     universe.loc[coin, "mark"], reg["risk_multiplier"] * edge_multiplier(model) * 0.5,
                     "3. SPECULATIVE FUTURES", f"Perpetual futures (e.g. Hyperliquid), isolated margin, <= {cfg.max_leverage:.0f}x",
                     cfg.equity * cfg.max_leverage * 0.25, news_ctx, tv)
    t.rationale.insert(1, f"Strongest Asia-session buying among mid-caps (z {session_z[coin]:+.2f})")
    t.extra["margin_at_max_lev"] = t.notional_usd / cfg.max_leverage
    t.extra["funding_apr"] = float(universe.loc[coin, "funding_1h"] * 24 * 365)
    return t


def pick_stablecoin(cfg, reg, carry: pd.DataFrame, xvenue: pd.DataFrame, hlp: float) -> Trade:
    """Best price-neutral use of stablecoins this week, or hold."""
    notional = cfg.equity * 0.30 * reg["risk_multiplier"]
    if len(carry) and carry.iloc[0]["net_apr"] >= cfg.carry_min_net_apr and carry.iloc[0]["positive_days"] >= 0.8:
        coin, r = carry.index[0], carry.iloc[0]
        return Trade("2. STABLECOIN CARRY", coin, "LONG SPOT + SHORT PERP (1x, delta-neutral)",
                     "Spot on any CEX + short perp on Hyperliquid, USDC collateral", float("nan"), float("nan"),
                     float("nan"), 0.0, notional, 0.0, 7, "MEDIUM",
                     [f"14-day average funding {r['avg14_apr']:.0%} APR (today {r['today_apr']:.0%}), "
                      f"positive {r['positive_days']:.0%} of days",
                      f"After fees ~{r['net_apr']:.0%} APR, ~${r['net_apr'] / 52 * notional:,.0f}/week per ${notional:,.0f} leg",
                      "Exit if funding drops under 10% APR; keep >=50% margin on the short"],
                     {"tradingview": tradingview_link(coin)}, {"net_apr": float(r["net_apr"])})
    ok = xvenue[(xvenue["persistent"]) & (xvenue["net_apr"] >= cfg.carry_min_net_apr)] if len(xvenue) else xvenue
    if len(ok):
        coin, r = ok.index[0], ok.iloc[0]
        hi, lo = ("Hyperliquid", r["other_venue"]) if r["gap_apr"] > 0 else (r["other_venue"], "Hyperliquid")
        return Trade("2. STABLECOIN CROSS-EXCHANGE FUNDING", coin, f"SHORT PERP on {hi} + LONG PERP on {lo}",
                     "Two perp exchanges, USDC/USDT collateral on both", float("nan"), float("nan"), float("nan"),
                     0.0, notional, 0.0, 7, "LOW-MEDIUM (other venue has no funding history here)",
                     [f"Funding gap {abs(r['gap_apr']):.0%} APR (Hyperliquid {r['hl_apr']:.0%}, "
                      f"{r['other_venue']} {r['other_apr']:.0%}); Hyperliquid 14-day avg {r['hl_avg14_apr']:.0%}",
                      f"After fees ~{r['net_apr']:.0%} APR",
                      "Needs accounts on both exchanges; rebalance margin if price moves >20%"],
                     {"tradingview": tradingview_link(coin)}, {"net_apr": float(r["net_apr"])})
    return Trade("2. STABLECOIN", "USDC", "HOLD (no price-neutral trade clears fees)", "Your exchange or wallet",
                 float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, 7, "n/a",
                 [f"Best spot-perp carry after fees: {carry.iloc[0]['net_apr']:.0%} APR ({carry.index[0]})"
                  if len(carry) else "No carry candidates",
                  f"Hyperliquid HLP vault: {hlp:.1%} APR (market-making vault; can lose money in crashes)"
                  if hlp == hlp else "HLP vault APR unavailable",
                  "Plain USDC lending on a major exchange or Aave is the low-risk default"], {}, {})
