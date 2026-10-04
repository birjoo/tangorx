# simons-crypto — a Renaissance-style weekly crypto trader

Generates **3 trades every week** (core spot, futures, speculative DEX) from a
statistical model that validates itself out-of-sample before it is allowed to trade.

> Not financial advice. Paper-trade for 8+ weeks and check `evaluate` before using real money.

## How Jim Simons would approach crypto

| Simons / Renaissance principle | How this engine applies it |
|---|---|
| **Data over narratives.** Medallion hired mathematicians, not market storytellers. | Direction comes only from price, volume and funding data. News is a *kill switch* (hacks, exploits, delistings, lawsuits, unlocks), never a reason to buy. |
| **Many weak signals, combined.** Each edge is tiny (Renaissance said they were "right 50.75% of the time"). | 8 signals (momentum 28d/7d, 3-day reversal, 50d trend, low-vol, volume surge, funding carry, close location) are z-scored across coins and blended. |
| **Let the data choose the weights and even the sign.** | Each signal's rank IC vs next-week return is measured on non-overlapping weeks. Weight ∝ information ratio. Anything with \|t\| < 1 gets zero weight. A signal that works in reverse gets a negative weight. |
| **No look-ahead, and paranoia about overfitting.** | Walk-forward: weights at week *t* only use ICs whose forward window had closed. The combined model's own out-of-sample IC and long/short Sharpe are reported every week. If it fails, the report says **NOT VALIDATED** and tells you to halve size or paper-trade. |
| **Market-neutral where possible.** Medallion mostly avoided betting on market direction. | The futures slot prefers a **delta-neutral funding-carry** trade (long spot + short perp, stablecoin collateral) when funding is ≥25% APR and positive on ≥80% of the last 14 days. |
| **Short horizons, high turnover, strict exits.** | 7-day holding period, ATR-based stop and target, everything closed and re-decided each week. |
| **Risk first.** Size by volatility, never by conviction alone. | Each trade risks a fixed % of equity at its stop (1% core, 1% futures, 0.25% speculative). All sizes are halved when BTC is below its 50-day average, and cut further at Fear & Greed ≥ 80. Spot positions are capped at 25% of equity, leverage at 3x. |
| **Measure everything.** | Every live pick is logged to `reports/journal.csv`. `python run_weekly.py evaluate` scores stop/target/time outcomes. |

## The three weekly slots

1. **Core spot (long only).** The top-ranked coin on the composite score among the 25 most liquid assets, skipped if it has risk headlines.
2. **Futures.** Either a market-neutral funding carry (stablecoin yield) or the strongest long *or short* perp signal. Isolated margin, ≤3x.
3. **Speculative DEX.** DexScreener tokens filtered hard for rug risk: liquidity ≥ $150k, 24h volume ≥ $250k, age ≥ 7 days, FDV/liquidity ≤ 60, volume/liquidity ≤ 25 (above that looks like wash trading), buyers ≥ 50%, ≥ 300 txns. Ranked on momentum, buy pressure, volume acceleration and liquidity. **Paid DexScreener boosts count against a token** (contrarian). This slot cannot be backtested with free data, so it gets lottery-ticket sizing.

The report also includes a funding-carry board, a stablecoin depeg monitor, TradingView chart links, and the learned signal weights.

## Data sources (free, no API keys)

- **Hyperliquid API**: perp universe, daily candles, hourly funding, open interest. Hyperliquid has no geo-block, so it also runs from US GitHub runners where Binance is blocked.
- **DexScreener API**: boosts, token profiles, pair data.
- **TradingView scanner**: technical rating, used for context only.
- **RSS** (CoinDesk, Cointelegraph, Decrypt): headline kill switch.
- **alternative.me**: Fear & Greed regime.

## Simplest: the web page

Open `web/index.html` in any browser (double-click it). Your browser pulls live data from Hyperliquid and DexScreener and shows the 3 trades sized to your account. You don't need to install anything. Press **Refresh** each Monday.

## Python usage (full report, journal, evaluation)

```bash
cd simons-crypto
pip install -r requirements.txt
python run_weekly.py --demo              # offline, synthetic data
python run_weekly.py --equity 10000      # live: writes reports/YYYY-Www.md + .json, appends journal
python run_weekly.py evaluate            # score past picks
python -m pytest -q tests
```

Run it on Sunday or Monday. A live run takes about 3–6 minutes (60 coins × 400 days of candles and funding).

### Automate weekly

Copy `github-workflow-weekly.yml` to `.github/workflows/` in a **separate private repo**. It runs every Monday and commits the report.

## Honest limitations

- Crypto has about 6 years of liquid perp history and regimes change. Expect the out-of-sample IC to decay. Trust the verdict line.
- Daily bars cannot tell whether the stop or the target was hit first within a day. `evaluate` assumes the stop (conservative).
- The DEX slot and the news filter are heuristics, not validated models.
- Renaissance's real edge came from execution, data cleaning and decades of research. This is the philosophy at retail scale, not Medallion.
