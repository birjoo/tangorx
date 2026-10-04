import math

import numpy as np
import pandas as pd

from quant.backtest import build_model
from quant.news import blocked, coin_news
from quant.report import append_journal, render
from quant.signals import atr, compute_signals
from quant.strategy import (Config, carry_candidates, pick_core_spot, pick_futures, pick_speculative,
                            regime, score_dex)
from quant.synthetic import make_dex, make_market


def _setup():
    panel, universe = make_market()
    model = build_model(compute_signals(panel), panel["close"])
    return panel, universe, model


def test_signals_have_no_lookahead():
    panel, _ = make_market()
    cut = {k: v.iloc[:-30] for k, v in panel.items()}
    full = compute_signals(panel)
    part = compute_signals(cut)
    d = cut["close"].index[-1]
    for name in full:
        pd.testing.assert_series_equal(full[name].loc[d], part[name].loc[d], check_names=False)


def test_walk_forward_finds_planted_momentum_edge():
    _, _, model = _setup()
    assert model.trustworthy
    assert model.summary["oos_mean_ic"] > 0.03
    assert model.weights.loc["mom_28d", "weight"] > 0


def test_pure_noise_is_not_trusted():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=300, freq="D", tz="UTC")
    cols = [f"X{i}" for i in range(25)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.04, (300, 25)), axis=0)), index=idx, columns=cols)
    panel = {"close": close, "high": close * 1.02, "low": close * 0.98,
             "dollar_volume": close * 0 + 1e7, "funding": close * 0}
    m = build_model(compute_signals(panel), close)
    assert abs(m.summary["oos_mean_ic"]) < 0.05


def test_news_kill_switch():
    heads = pd.DataFrame({"title": ["Solana DEX exploited for $10M", "Bitcoin rallies", "C05 token delisted"]})
    news = coin_news(["SOL", "BTC", "C05"], heads)
    assert blocked("SOL", news) and blocked("C05", news) and not blocked("BTC", news)
    assert news.loc["BTC", "mentions"] == 1


def test_three_trades_with_correct_risk():
    panel, universe, model = _setup()
    cfg = Config(equity=10_000)
    reg = regime(panel["close"], pd.Series([50.0]))
    news = coin_news(list(panel["close"].columns), pd.DataFrame({"title": []}))
    a = atr(panel)
    tv = pd.Series(dtype=float)
    core = pick_core_spot(cfg, model, universe, panel, a, reg, news, tv)
    fut = pick_futures(cfg, model, universe, panel, a, reg, news, tv, {core.symbol})
    spec = pick_speculative(cfg, score_dex(cfg, make_dex(200)), reg)
    assert core.side == "LONG" and core.stop < core.entry < core.target
    assert math.isclose(core.risk_usd, min(100 * reg["risk_multiplier"], core.risk_usd), rel_tol=1e-9)
    assert core.notional_usd <= cfg.equity * cfg.max_spot_alloc + 1e-6
    assert fut.symbol != core.symbol
    assert spec.risk_usd <= cfg.equity * cfg.risk_spec + 1e-9
    md = render(panel["close"].index[-1], [core, fut, spec], model, reg, cfg, pd.DataFrame(),
                carry_candidates(cfg, universe), pd.DataFrame(), demo=True)
    assert md.count("### ") == 3


def test_directional_futures_when_no_carry():
    panel, universe, model = _setup()
    universe = universe.assign(funding_1h=0.0)
    cfg = Config()
    reg = regime(panel["close"], pd.Series(dtype=float))
    news = coin_news(list(panel["close"].columns), pd.DataFrame({"title": []}))
    t = pick_futures(cfg, model, universe, panel, atr(panel), reg, news, pd.Series(dtype=float), set())
    assert "directional" in t.bucket
    if t.side == "SHORT":
        assert t.target < t.entry < t.stop


def test_dex_rug_filters():
    d = make_dex(200)
    s = score_dex(Config(), d)
    assert (s["liquidity_usd"] >= 150_000).all()
    assert (s["age_days"] >= 7).all()
    assert (s["buy_ratio"] >= 0.5).all()
    assert (s["vol_to_liq"] <= 25).all()


def test_journal(tmp_path):
    panel, universe, model = _setup()
    cfg = Config()
    reg = regime(panel["close"], pd.Series(dtype=float))
    news = coin_news(list(panel["close"].columns), pd.DataFrame({"title": []}))
    core = pick_core_spot(cfg, model, universe, panel, atr(panel), reg, news, pd.Series(dtype=float))
    p = tmp_path / "j.csv"
    append_journal(p, pd.Timestamp("2026-10-04"), [core])
    append_journal(p, pd.Timestamp("2026-10-11"), [core])
    assert len(pd.read_csv(p)) == 2
