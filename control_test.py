"""
Control test: does the factor RANKING add anything beyond just the
regime exposure filter applied to an equal-weighted basket of all 49
stocks? If this control lands close to the factor strategy's numbers,
the factor selection isn't earning its keep -- the regime filter is
still doing most of the work.
"""

import numpy as np
import pandas as pd

from data_loader import load_price_data, load_nifty
from portfolio.risk_filters import market_exposure

price_data = load_price_data()
nifty = load_nifty()

closes = pd.DataFrame({t: df["Close"] for t, df in price_data.items()}).sort_index().dropna(how="all")
closes = closes.ffill().dropna()
rets = closes.pct_change().dropna()
equal_weight_ret = rets.mean(axis=1)

exposures = pd.Series(
    [market_exposure(d, nifty) for d in equal_weight_ret.index],
    index=equal_weight_ret.index
).shift(1).fillna(1.0)

scaled_ret = equal_weight_ret * exposures
equity = (1 + scaled_ret).cumprod() * 100000

cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
vol = scaled_ret.std() * np.sqrt(252)
sharpe = cagr / vol
max_dd = (equity - equity.cummax()).min() / equity.cummax().max()

print("--- CONTROL: Equal-weight (all 49 stocks) + regime filter only, no factor ranking ---")
print(f"CAGR:         {cagr:>8.2%}")
print(f"Volatility:   {vol:>8.2%}")
print(f"Sharpe:       {sharpe:>8.2f}")
print(f"Max Drawdown: {max_dd:>8.2%}")