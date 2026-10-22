"""Compare signal sets out-of-sample on the cached live panel: is the new data worth anything?"""
import glob
import pickle
import sys

import pandas as pd

from quant.backtest import build_model
from quant.signals import V2_SIGNALS, compute_signals

path = sorted(glob.glob(".cache/panel-*.pkl"))[-1]
panel = pickle.load(open(path, "rb"))
sig = compute_signals(panel)
v1 = [k for k in sig if k not in V2_SIGNALS]
sets = {"v1 (old)": v1, "v2 (new only)": [k for k in V2_SIGNALS if k in sig], "all": list(sig)}
rows = {}
for name, keys in sets.items():
    m = build_model({k: sig[k] for k in keys}, panel["close"])
    rows[name] = {**m.summary, "trusted": m.trustworthy}
    if name == "all":
        weights = m.weights
print(pd.DataFrame(rows).T.round(3).to_string())
print()
print(weights.sort_values("t_stat", key=abs, ascending=False).round(3).to_string())
