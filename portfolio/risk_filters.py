import pandas as pd
import numpy as np


def market_exposure(date, benchmark, fast=50, slow=200):
    """
    Soft risk-off exposure scaler:
    - 1.0 → full exposure
    - 0.5 → defensive
    - 0.0 → capital preservation
    """

    # --- Handle Series or DataFrame ---
    if isinstance(benchmark, pd.Series):
        close = benchmark.loc[:date]
    else:
        close = benchmark.loc[:date, "Close"]

    # --- Not enough data ---
    if len(close) < slow:
        return 1.0

    ma_fast = close.rolling(fast).mean().iloc[-1]
    ma_slow = close.rolling(slow).mean().iloc[-1]
    price = close.iloc[-1]

    # --- Exposure logic ---
    if price > ma_fast > ma_slow:
        return 1.0        # strong uptrend
    elif price > ma_slow:
        return 0.6        # weak uptrend / chop
    else:
        return 0.2        # defensive mode
