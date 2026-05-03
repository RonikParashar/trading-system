# portfolio/slippage.py

import pandas as pd
import numpy as np

def estimate_slippage(
    returns: pd.Series,
    trade_fraction: float,
    base_spread: float = 0.0003,   # 3 bps
    vol_multiplier: float = 0.5,
    max_slippage: float = 0.005    # 50 bps HARD CAP
):
    """
    Realistic slippage model:
    - Applied only on traded fraction
    - Capped to avoid runaway losses
    """

    vol = returns.rolling(20).std().iloc[-1]
    if pd.isna(vol):
        vol = 0.0

    slip = base_spread + vol_multiplier * vol
    slip *= trade_fraction

    return min(slip, max_slippage)
