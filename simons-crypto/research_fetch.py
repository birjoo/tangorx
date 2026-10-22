"""Download and cache the research panel (60 most-traded perps, 400 days)."""
from quant import data
uni = data.perp_universe()
coins = uni.sort_values("volume_24h_usd", ascending=False).head(60).index.tolist()
p = data.load_panel(coins, 400, cache_dir=".cache")
print({k: v.shape for k, v in p.items()})
