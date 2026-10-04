"""Headline processing.

Simons famously ignored narratives. Here news is used only two ways:
  1. A kill switch: a coin with fresh hack/exploit/delisting/lawsuit headlines is skipped
     (tail risk the price model cannot see).
  2. An 'attention' count reported for context, never used to pick direction.
"""
from __future__ import annotations

import re

import pandas as pd

RISK_WORDS = re.compile(
    r"\b(hack(ed)?|exploit(ed)?|drain(ed)?|rug ?pull|delist(s|ed|ing)?|lawsuit|sued|sues|charged|"
    r"indict(ed|ment)?|bankrupt(cy)?|insolven(t|cy)|halt(s|ed)?|depeg(ged)?|outage|vulnerabilit(y|ies)|"
    r"stolen|fraud|seized?|attack(ed)?|unlock(s)?)\b", re.I)

NAMES = {
    "BTC": "bitcoin", "ETH": "ethereum|ether", "SOL": "solana", "XRP": "xrp|ripple", "BNB": "bnb|binance coin",
    "DOGE": "dogecoin", "ADA": "cardano", "AVAX": "avalanche", "LINK": "chainlink", "DOT": "polkadot",
    "TON": "toncoin", "SUI": "sui", "APT": "aptos", "ARB": "arbitrum", "OP": "optimism", "NEAR": "near protocol",
    "LTC": "litecoin", "TRX": "tron", "HYPE": "hyperliquid", "PEPE": "pepe", "WIF": "dogwifhat",
    "AAVE": "aave", "UNI": "uniswap", "ENA": "ethena", "TIA": "celestia", "INJ": "injective", "SEI": "sei",
}


def _pattern(coin: str) -> re.Pattern:
    alts = [re.escape(coin)] + ([NAMES[coin]] if coin in NAMES else [])
    return re.compile(r"\b(" + "|".join(alts) + r")\b", re.I if coin in NAMES else 0)


def coin_news(coins: list[str], heads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    titles = heads["title"].tolist() if len(heads) else []
    for coin in coins:
        if len(coin) < 3 and coin not in NAMES:
            continue  # two-letter tickers match too much noise
        pat = _pattern(coin)
        hits = [t for t in titles if pat.search(t)]
        risk = [t for t in hits if RISK_WORDS.search(t)]
        rows.append({"coin": coin, "mentions": len(hits), "risk_headlines": risk})
    return pd.DataFrame(rows, columns=["coin", "mentions", "risk_headlines"]).set_index("coin")


def blocked(coin: str, news: pd.DataFrame) -> list[str]:
    if coin in news.index:
        return list(news.loc[coin, "risk_headlines"])
    return []
