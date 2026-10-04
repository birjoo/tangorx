# Weekly Quant Picks — 2026-10-04 (week 2026-W40)

> **DEMO RUN on synthetic data. Not real prices. Do not trade this.**

> Systematic, statistical, small edges, strict risk. Not financial advice. Paper-trade first.

## Regime

- BTC above 50-day average: **no**
- BTC 30d realised vol (ann.): 144%
- Fear & Greed: 55
- Risk multiplier applied to all sizes: **0.50x**

## Model health (walk-forward, out-of-sample)

- Weeks tested: 44
- Mean rank IC: +0.091  (0.02–0.05 is a real edge at scale)
- IC hit rate: 70%
- Top-minus-bottom quintile Sharpe (ann.): 3.01
- Verdict: **TRADEABLE**

## This week's 3 trades (equity $10,000)

### 1. CORE SPOT: C18 — LONG

| Field | Value |
|---|---|
| Venue | Any major spot exchange (paid in USDC/USDT) |
| Entry | 9.5984 |
| Stop | 7.7380 |
| Target | 12.3890 |
| Size | 26.88 units ($258 notional) |
| Max loss at stop | $50 |
| Horizon | 7 days (exit at market if neither stop nor target hit) |
| Conviction | HIGH |

- Composite z-score +1.77 (rank 1/30)
- Dominant validated signals: trend_50d, mom_28d, reversal_3d
- Stop 2.0xATR, target 3.0xATR, 7-day time stop
- 0 headlines this week (context only)
- [tradingview](https://www.tradingview.com/chart/?symbol=BINANCE:C18USDT)

### 2. FUTURES (market-neutral carry): C02 — LONG SPOT + SHORT PERP (1x, delta-neutral)

| Field | Value |
|---|---|
| Venue | Spot on any CEX + short perp on Hyperliquid, USDC collateral |
| Entry | 1.4069 |
| Stop | n/a |
| Target | n/a |
| Size | 1,066 units ($1,500 notional) |
| Max loss at stop | $0 |
| Horizon | 7 days (exit at market if neither stop nor target hit) |
| Conviction | MEDIUM |

- Current funding 44% APR, positive on 100% of last 14 days
- Expected funding income ~$13/week on $1,500 per leg
- Price-neutral: profit is the funding longs pay you; exit if APR < 10% or flips negative
- Risks: funding flip, exchange risk, liquidation of perp leg if under-margined (keep >=50% margin)
- [tradingview](https://www.tradingview.com/chart/?symbol=BINANCE:C02USDT)

### 3. SPECULATIVE (DEX): MEME16 (solana) — LONG

| Field | Value |
|---|---|
| Venue | raydium on solana |
| Entry | 0.00391989 |
| Stop | 0.00293992 |
| Target | 0.00627182 |
| Size | 1.276e+04 units ($50 notional) |
| Max loss at stop | $12 |
| Horizon | 7 days (exit at market if neither stop nor target hit) |
| Conviction | LOW (unvalidated, lottery-ticket sizing) |

- Liquidity $478,463, 24h vol $7,474,372, age 44d
- 24h +7%, 6h +2%, buy ratio 82%
- FDV/liquidity 12.7x, vol/liquidity 15.6x
- Passed rug filters; ranked #1 of 3 survivors
- Verify contract on a token scanner (honeypot/mint authority) before buying
- Use limit orders, max 1% slippage; size so a total loss is irrelevant
- [dexscreener](https://dexscreener.com/solana/addr16)

## Signal weights (learned from data, not opinion)

| Signal | Weeks | Mean IC | t-stat | Weight |
|---|---|---|---|---|
| trend_50d | 52 | +0.092 | +3.61 | +0.26 |
| mom_28d | 53 | +0.076 | +3.23 | +0.23 |
| reversal_3d | 54 | -0.059 | -2.18 | -0.15 |
| low_vol | 54 | -0.051 | -1.88 | -0.13 |
| close_location | 56 | -0.039 | -1.77 | -0.12 |
| mom_7d | 54 | +0.038 | +1.42 | +0.10 |
| volume_surge | 55 | +0.005 | +0.18 | +0.00 |
| funding_carry | 56 | +0.008 | +0.35 | +0.00 |

## Top / bottom of the composite ranking

Longs: C18 (+1.77), C08 (+1.03), C06 (+0.81), C09 (+0.76), ETH (+0.70), C01 (+0.64), C03 (+0.53), C04 (+0.46)

Shorts: SOL (-1.18), C19 (-0.88), C26 (-0.63), C15 (-0.61), C10 (-0.49), C17 (-0.46), C07 (-0.44), C16 (-0.43)

## Funding carry board (stablecoin yield via delta-neutral)

| Perp | Funding APR | 24h volume |
|---|---|---|
| C02 | +44% | $91M |
| SOL | +35% | $69M |
| C03 | +28% | $130M |
| C01 | +26% | $276M |
| BTC | +25% | $87M |
| C08 | +24% | $63M |

## Headlines scanned (kill-switch input only)

- C03 protocol exploited for $40M
- Bitcoin ETF inflows rise

## Rules

1. Enter within 24h of the report, limit orders only.
2. Stops are placed immediately. Never moved further away.
3. Exit everything after 7 days; next report re-decides from scratch.
4. Log fills in `reports/journal.csv`; run `python run_weekly.py evaluate` to score the system.
5. If total drawdown hits 15%, stop and re-validate the model.
