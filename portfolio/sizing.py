import pandas as pd
import numpy as np

def compute_position_size(df: pd.DataFrame,
                          equity: float,
                          risk_per_trade: float = 0.01,
                          max_position_pct: float = 0.15) -> pd.Series:
    """
    Computes position value per day using volatility targeting.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for one stock
    equity : float
        Current portfolio equity
    risk_per_trade : float
        Fraction of equity to risk per trade
    max_position_pct : float
        Max fraction of equity in a single position

    Returns
    -------
    pd.Series
        Position value per day (₹)
    """

    returns = df["Close"].pct_change()
    volatility = returns.rolling(20).std()

    # raw position size
    position_value = (equity * risk_per_trade) / volatility

    # cap max position size
    max_value = equity * max_position_pct
    position_value = position_value.clip(upper=max_value)

    return position_value.replace([np.inf, -np.inf], 0).fillna(0)
