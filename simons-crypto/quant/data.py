"""Data sources. All free, public, no API keys.

- Hyperliquid  : perp universe, daily candles, funding, open interest (no geo-block)
- DexScreener  : on-chain DEX pairs for the speculative bucket
- TradingView  : unofficial scanner endpoint, used only as one weak feature
- News RSS     : headline attention + risk-word kill switch
- alternative.me Fear & Greed : regime feature
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

HL_INFO = "https://api.hyperliquid.xyz/info"
DEX = "https://api.dexscreener.com"
TV_SCAN = "https://scanner.tradingview.com/crypto/scan"
FNG = "https://api.alternative.me/fng/?limit=30"
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

DEX_SEARCH_QUERIES = ["SOL", "WETH", "WBNB", "USDC", "meme", "AI", "base", "pump", "bonk", "virtual"]
STABLECOINS = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD", "USDD", "FRAX", "BUSD", "USDS", "RLUSD"}

_session = requests.Session()
_session.headers["User-Agent"] = "simons-crypto/1.0"


def _post(url: str, payload: dict, retries: int = 4):
    for i in range(retries):
        try:
            r = _session.post(url, json=payload, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def _get(url: str, retries: int = 4, as_json: bool = True):
    for i in range(retries):
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json() if as_json else r.text
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


# ---------------------------------------------------------------- Hyperliquid
def perp_universe() -> pd.DataFrame:
    """Snapshot of every Hyperliquid perp: mark, funding (hourly), OI, 24h volume."""
    meta, ctxs = _post(HL_INFO, {"type": "metaAndAssetCtxs"})
    rows = []
    for asset, ctx in zip(meta["universe"], ctxs):
        if asset.get("isDelisted"):
            continue
        mark = float(ctx.get("markPx") or 0)
        rows.append({
            "coin": asset["name"],
            "max_leverage": asset.get("maxLeverage"),
            "mark": mark,
            "funding_1h": float(ctx.get("funding") or 0),
            "open_interest_usd": float(ctx.get("openInterest") or 0) * mark,
            "volume_24h_usd": float(ctx.get("dayNtlVlm") or 0),
            "prev_day_px": float(ctx.get("prevDayPx") or 0),
        })
    df = pd.DataFrame(rows).set_index("coin")
    return df[~df.index.isin(STABLECOINS)]


def daily_candles(coin: str, days: int) -> pd.DataFrame:
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86_400_000
    raw = _post(HL_INFO, {"type": "candleSnapshot",
                          "req": {"coin": coin, "interval": "1d", "startTime": start, "endTime": end}})
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.normalize()
    out = df.set_index("date")[["o", "h", "l", "c", "v"]].astype(float)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out


def funding_history(coin: str, days: int) -> pd.Series:
    """Daily mean of hourly funding rates."""
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: list[dict] = []
    while True:
        batch = _post(HL_INFO, {"type": "fundingHistory", "coin": coin, "startTime": start})
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["time"]
        if len(batch) < 500 or last <= start:
            break
        start = last + 1
        time.sleep(0.2)
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.normalize()
    return df.assign(r=df["fundingRate"].astype(float)).groupby("date")["r"].mean()


def load_panel(coins: list[str], days: int, with_funding: bool = True, funding_days: int = 120) -> dict[str, pd.DataFrame]:
    """Wide panels (date x coin) for close/high/low/volume/funding."""
    closes, highs, lows, vols, funds = {}, {}, {}, {}, {}
    for i, coin in enumerate(coins):
        c = daily_candles(coin, days)
        if len(c) < 60:
            continue
        closes[coin], highs[coin], lows[coin], vols[coin] = c["close"], c["high"], c["low"], c["volume"] * c["close"]
        if with_funding:
            funds[coin] = funding_history(coin, min(days, funding_days))
        time.sleep(0.15)
    panel = {k: pd.DataFrame(v).sort_index() for k, v in
             dict(close=closes, high=highs, low=lows, dollar_volume=vols).items()}
    panel["funding"] = (pd.DataFrame(funds).reindex(panel["close"].index) if funds
                        else pd.DataFrame(0.0, index=panel["close"].index, columns=panel["close"].columns))
    return panel


# ---------------------------------------------------------------- DexScreener
def dex_candidates(max_tokens: int = 240) -> pd.DataFrame:
    """Discovery via DexScreener boosts + latest profiles, then full pair data."""
    seen: dict[tuple[str, str], float] = {}
    for path in ("/token-boosts/top/v1", "/token-boosts/latest/v1", "/token-profiles/latest/v1"):
        try:
            for t in _get(DEX + path) or []:
                key = (t.get("chainId"), t.get("tokenAddress"))
                if all(key):
                    seen[key] = max(seen.get(key, 0.0), float(t.get("totalAmount") or t.get("amount") or 0))
        except requests.RequestException:
            continue
    # Boosts/profiles skew to brand-new launches; add established liquid pairs via search
    for q in DEX_SEARCH_QUERIES:
        try:
            for p in (_get(f"{DEX}/latest/dex/search?q={q}") or {}).get("pairs") or []:
                key = (p.get("chainId"), p["baseToken"]["address"])
                if (p.get("liquidity") or {}).get("usd", 0) >= 100_000 and key not in seen:
                    seen[key] = 0.0
        except requests.RequestException:
            continue
    by_chain: dict[str, list[str]] = {}
    for chain, addr in list(seen)[:max_tokens]:
        by_chain.setdefault(chain, []).append(addr)

    rows = []
    for chain, addrs in by_chain.items():
        for i in range(0, len(addrs), 30):
            try:
                pairs = _get(f"{DEX}/tokens/v1/{chain}/{','.join(addrs[i:i + 30])}") or []
            except requests.RequestException:
                continue
            best: dict[str, dict] = {}
            for p in pairs:  # keep most liquid pair per token
                addr = p["baseToken"]["address"]
                if (p.get("liquidity") or {}).get("usd", 0) > (best.get(addr, {}).get("liquidity") or {}).get("usd", -1):
                    best[addr] = p
            for addr, p in best.items():
                rows.append(flatten_dex_pair(p, boost=seen.get((chain, addr), 0.0)))
            time.sleep(0.3)
    return pd.DataFrame(rows)


def flatten_dex_pair(p: dict, boost: float = 0.0) -> dict:
    tx = (p.get("txns") or {}).get("h24") or {}
    pc = p.get("priceChange") or {}
    vol = p.get("volume") or {}
    return {
        "symbol": p["baseToken"]["symbol"],
        "name": p["baseToken"].get("name"),
        "chain": p.get("chainId"),
        "dex": p.get("dexId"),
        "address": p["baseToken"]["address"],
        "url": p.get("url"),
        "price_usd": float(p.get("priceUsd") or 0),
        "liquidity_usd": float((p.get("liquidity") or {}).get("usd") or 0),
        "fdv": float(p.get("fdv") or 0),
        "market_cap": float(p.get("marketCap") or 0),
        "vol_h24": float(vol.get("h24") or 0),
        "vol_h6": float(vol.get("h6") or 0),
        "vol_h1": float(vol.get("h1") or 0),
        "chg_h1": float(pc.get("h1") or 0),
        "chg_h6": float(pc.get("h6") or 0),
        "chg_h24": float(pc.get("h24") or 0),
        "buys_h24": int(tx.get("buys") or 0),
        "sells_h24": int(tx.get("sells") or 0),
        "age_days": ((time.time() * 1000 - p["pairCreatedAt"]) / 86_400_000) if p.get("pairCreatedAt") else 0.0,
        "boost": boost,
    }


def stablecoin_pegs() -> pd.DataFrame:
    """Most liquid DEX price for major stablecoins -> depeg monitor."""
    rows = []
    for sym in ("USDC", "USDT", "DAI", "USDe", "USDS", "PYUSD"):
        try:
            pairs = (_get(f"{DEX}/latest/dex/search?q={sym}%20USD") or {}).get("pairs") or []
        except requests.RequestException:
            continue
        pairs = sorted((p for p in pairs if p["baseToken"]["symbol"].upper() == sym.upper()
                        and (p.get("liquidity") or {}).get("usd", 0) > 1_000_000),
                       key=lambda x: -x["liquidity"]["usd"])[:5]
        if pairs:
            # median of the 5 deepest pools: one scam token sharing the ticker can't fake a depeg
            p = sorted(pairs, key=lambda x: float(x.get("priceUsd") or 0))[len(pairs) // 2]
            px = float(p.get("priceUsd") or 1)
            rows.append({"stable": sym, "price": px, "deviation_bps": (px - 1) * 1e4, "url": p.get("url")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- TradingView
def tradingview_ratings(coins: list[str]) -> pd.Series:
    """TradingView 'Recommend.All' (-1 strong sell .. +1 strong buy). Unofficial endpoint."""
    tickers = [f"BINANCE:{c}USDT" for c in coins]
    try:
        res = _post(TV_SCAN, {"symbols": {"tickers": tickers, "query": {"types": []}},
                              "columns": ["Recommend.All"]}, retries=2)
    except requests.RequestException:
        return pd.Series(dtype=float)
    out = {}
    for row in res.get("data", []):
        coin = row["s"].split(":")[1].removesuffix("USDT")
        if row["d"] and row["d"][0] is not None:
            out[coin] = float(row["d"][0])
    return pd.Series(out)


def tradingview_link(coin: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{coin}USDT"


# ---------------------------------------------------------------- News / sentiment
def headlines(days: int = 7) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for feed in NEWS_FEEDS:
        try:
            root = ET.fromstring(_get(feed, retries=2, as_json=False))
        except (requests.RequestException, ET.ParseError):
            continue
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            pub = item.findtext("pubDate")
            try:
                ts = pd.to_datetime(pub, utc=True)
            except (ValueError, TypeError):
                ts = pd.NaT
            if pd.isna(ts) or ts >= cutoff:
                rows.append({"title": title, "published": ts, "source": feed.split("/")[2]})
    return pd.DataFrame(rows, columns=["title", "published", "source"])


def fear_greed() -> pd.Series:
    try:
        data = _get(FNG)["data"]
    except (requests.RequestException, KeyError):
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s", utc=True): float(d["value"]) for d in data})
    return s.sort_index()
