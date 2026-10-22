"""Signal library.

Simons' rule: no single signal is trusted. Each one is a weak, noisy predictor
of next-week *relative* return. They are standardised cross-sectionally every
day, then combined with weights learned out-of-sample (see backtest.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 7  # days: we trade weekly


def _vol(close: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    return np.log(close).diff().rolling(n, min_periods=n // 2).std()


def zscore_xs(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per date, winsorised (fat tails are the norm in crypto)."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-clip, clip)


def compute_signals(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Raw signals, all oriented so that 'higher = expected to outperform' under the
    textbook hypothesis. The backtest is free to flip the sign if data disagrees."""
    close, vol_usd, fund = panel["close"], panel["dollar_volume"], panel["funding"]
    high, low = panel["high"], panel["low"]
    logp = np.log(close)
    vol30 = _vol(close)

    sig = {
        # Time-series momentum, skip last 3 days to separate from reversal
        "mom_28d": (logp.shift(3) - logp.shift(28)) / (vol30 * np.sqrt(25)),
        "mom_7d": (logp - logp.shift(7)) / (vol30 * np.sqrt(7)),
        # Short-term reversal: liquidity providers get paid for absorbing shocks
        "reversal_3d": -(logp - logp.shift(3)) / (vol30 * np.sqrt(3)),
        # Distance from 50d trend
        "trend_50d": (logp - logp.rolling(50, min_periods=30).mean()) / vol30,
        # Low-volatility anomaly
        "low_vol": -vol30,
        # Abnormal dollar volume (attention)
        "volume_surge": np.log(vol_usd.rolling(3).mean() / vol_usd.rolling(30, min_periods=15).mean()),
        # Funding carry: crowded longs pay funding and tend to underperform
        "funding_carry": -fund.rolling(7, min_periods=3).mean(),
        # Close location in daily range (buying pressure into the close)
        "close_location": (((close - low) / (high - low).replace(0, np.nan)) - 0.5).rolling(5).mean(),
    }
    sig.update(compute_signals_v2(panel))
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in sig.items()}


V2_SIGNALS = ["resid_rev_5d", "low_beta", "max_effect", "fip_28d", "funding_change", "skew_4h", "asia_minus_us"]


def compute_signals_v2(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Second-generation signals: market-residual stat-arb (Renaissance's bread and butter),
    lottery/skew effects, path quality of momentum, funding dynamics, and intraday session flow."""
    close, fund = panel["close"], panel["funding"]
    ret = np.log(close).diff()
    mkt = ret.mean(axis=1)  # equal-weight crypto market
    cov = ret.rolling(60, min_periods=30).cov(mkt)
    beta = cov.div(mkt.rolling(60, min_periods=30).var(), axis=0)
    resid = ret - beta.mul(mkt, axis=0)
    rvol = resid.rolling(30, min_periods=15).std()
    sig = {
        # Idiosyncratic moves (after removing the market) tend to mean-revert
        "resid_rev_5d": -resid.rolling(5).sum() / (rvol * np.sqrt(5)),
        # Low-beta coins earn more per unit of risk
        "low_beta": -beta,
        # Lottery effect: coins with one huge up-day are overbought
        "max_effect": -ret.rolling(30, min_periods=15).max(),
        # Frog-in-the-pan: momentum built from many small moves persists better than from jumps
        "fip_28d": np.sign(ret.rolling(28).sum()) * ((ret > 0).astype(float).rolling(28).mean()
                                                      - (ret < 0).astype(float).rolling(28).mean()),
        # Funding rising faster than its own norm = longs crowding in
        "funding_change": -(fund.rolling(7, min_periods=3).mean() - fund.rolling(30, min_periods=10).mean()),
    }
    c4 = panel.get("close_4h")
    if c4 is not None and not c4.empty:
        r4 = np.log(c4).diff()
        day = r4.index.normalize()
        # Negative skew of 4h returns over 14 days, sampled at day end
        sig["skew_4h"] = -r4.rolling(84, min_periods=42).skew().groupby(day).last().reindex(close.index)
        # Returns made in Asia hours (00-08 UTC) minus US hours (12-20 UTC), last 14 days
        hrs = pd.Series(r4.index.hour, index=r4.index)
        asia = r4.mul(((hrs >= 0) & (hrs < 8)).astype(float), axis=0).groupby(day).sum()
        us = r4.mul(((hrs >= 12) & (hrs < 20)).astype(float), axis=0).groupby(day).sum()
        sig["asia_minus_us"] = (asia - us).rolling(14, min_periods=7).sum().reindex(close.index)
    return sig


def forward_returns(close: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Forward log return, demeaned cross-sectionally (we predict relative performance)."""
    fwd = np.log(close.shift(-horizon) / close)
    return fwd.sub(fwd.mean(axis=1), axis=0)


def atr(panel: dict[str, pd.DataFrame], n: int = 14) -> pd.DataFrame:
    close, high, low = panel["close"], panel["high"], panel["low"]
    prev = close.shift(1)
    tr = np.fmax(np.fmax(high - low, (high - prev).abs()), (low - prev).abs())
    return tr.rolling(n, min_periods=n // 2).mean()
