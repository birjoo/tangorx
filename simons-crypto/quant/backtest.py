"""Walk-forward signal validation and weighting.

Renaissance-style discipline:
  * Measure every signal's predictive power (rank IC) on non-overlapping weeks.
  * Only use information available at the time (no look-ahead).
  * Weight by information ratio, drop anything statistically indistinguishable from noise.
  * Judge the *combined* model purely out-of-sample before trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import HORIZON, forward_returns, zscore_xs

MIN_COINS = 8
MIN_OBS = 12      # weeks of IC history before a signal can earn weight
T_MIN = 1.0       # |t-stat| gate; deliberately loose because we combine many weak signals


def rank_ic(sig: pd.DataFrame, fwd: pd.DataFrame, dates) -> pd.Series:
    out = {}
    for d in dates:
        a, b = sig.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= MIN_COINS:
            out[d] = a[m].rank().corr(b[m].rank())
    return pd.Series(out, dtype=float)


def weights_from_ics(ics: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, s in ics.items():
        s = s.dropna()
        n = len(s)
        mean, std = (s.mean(), s.std()) if n > 1 else (np.nan, np.nan)
        t = mean / std * np.sqrt(n) if n > 1 and std > 0 else 0.0
        ir = mean / std if n > 1 and std > 0 else 0.0
        rows[name] = {"n_weeks": n, "mean_ic": mean, "ic_std": std, "t_stat": t,
                      "raw_weight": ir if (n >= MIN_OBS and abs(t) >= T_MIN) else 0.0}
    df = pd.DataFrame(rows).T
    total = df["raw_weight"].abs().sum()
    df["weight"] = df["raw_weight"] / total if total > 0 else 0.0
    return df.drop(columns="raw_weight")


@dataclass
class ModelReport:
    weights: pd.DataFrame          # current signal stats & weights
    oos_ic: pd.Series              # composite out-of-sample IC per week
    oos_long_short: pd.Series      # weekly top-minus-bottom quintile return (OOS)
    composite_today: pd.Series     # latest composite score per coin

    @property
    def summary(self) -> dict:
        ic, ls = self.oos_ic.dropna(), self.oos_long_short.dropna()
        ann = np.sqrt(52)
        return {
            "oos_weeks": int(len(ic)),
            "oos_mean_ic": float(ic.mean()) if len(ic) else float("nan"),
            "oos_ic_hit_rate": float((ic > 0).mean()) if len(ic) else float("nan"),
            "oos_ls_sharpe": float(ls.mean() / ls.std() * ann) if len(ls) > 2 and ls.std() > 0 else float("nan"),
            "oos_ls_weekly_mean": float(ls.mean()) if len(ls) else float("nan"),
            "recent_ls_sharpe": (float(ls.tail(26).mean() / ls.tail(26).std() * ann)
                                 if len(ls) > 10 and ls.tail(26).std() > 0 else float("nan")),
        }

    @property
    def trustworthy(self) -> bool:
        s = self.summary
        return (s["oos_weeks"] >= 8 and s["oos_mean_ic"] > 0.0 and s["oos_ic_hit_rate"] >= 0.5
                and s["oos_ls_sharpe"] > 0.0)


# Fixed-weight model adopted 2026-10-22 after the v2 research (research.py): one defensive
# factor (three correlated signals sharing a third of the weight each) plus Asia-session flow.
# Fixed weights are not fitted to the data, which limits overfitting.
FIXED_V2 = {"low_vol": 1 / 6, "low_beta": 1 / 6, "max_effect": 1 / 6, "asia_minus_us": 1 / 2}


def build_model(signals: dict[str, pd.DataFrame], close: pd.DataFrame,
                horizon: int = HORIZON, fixed: dict[str, float] | None = None) -> ModelReport:
    if fixed:
        fixed = {k: w for k, w in fixed.items() if k in signals}  # e.g. no 4h bars in demo data
        tot = sum(abs(w) for w in fixed.values()) or 1.0
        fixed = {k: w / tot for k, w in fixed.items()}
        signals = {k: v for k, v in signals.items() if k in fixed}
    fwd = forward_returns(close, horizon)
    z = {k: zscore_xs(v) for k, v in signals.items()}
    idx = close.index
    # Non-overlapping weekly sample dates, anchored to the most recent date
    dates = list(idx[::-1][::horizon][::-1])
    scored = [d for d in dates if d <= idx[-1 - horizon]] if len(idx) > horizon else []

    ics = {k: rank_ic(v, fwd, scored) for k, v in z.items()}

    # Walk-forward composite: weights at date d only use ICs whose forward window closed before d
    oos_ic, oos_ls = {}, {}
    for d in scored:
        known = {k: (s[s.index <= d - pd.Timedelta(days=horizon)] if len(s) else s) for k, s in ics.items()}
        w = pd.Series(fixed) if fixed else weights_from_ics(known)["weight"]
        if w.abs().sum() == 0:
            continue
        comp = sum(z[k].loc[d].fillna(0) * w[k] for k in w.index)
        r = fwd.loc[d]
        m = comp.notna() & r.notna() & close.loc[d].notna()
        if m.sum() < MIN_COINS:
            continue
        c, rr = comp[m], r[m]
        oos_ic[d] = c.rank().corr(rr.rank())
        q = max(1, len(c) // 5)
        order = c.sort_values().index
        oos_ls[d] = rr[order[-q:]].mean() - rr[order[:q]].mean()

    final_w = weights_from_ics(ics)
    if fixed:
        final_w["weight"] = pd.Series(fixed)
    last = idx[-1]
    comp_today = sum(z[k].loc[last].fillna(0) * final_w.loc[k, "weight"] for k in final_w.index)
    comp_today = comp_today[close.loc[last].notna()].dropna().sort_values(ascending=False)
    return ModelReport(final_w, pd.Series(oos_ic, dtype=float), pd.Series(oos_ls, dtype=float), comp_today)
