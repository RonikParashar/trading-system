import pandas as pd
import numpy as np

def compute_score(df: pd.DataFrame) -> pd.Series:
    """
    Continuous score for ranking stocks.
    Higher = better opportunity.
    """

    close = df["Close"]

    # long-term trend
    ma_200 = close.rolling(200).mean()
    trend_strength = (close - ma_200) / ma_200

    # medium-term momentum
    momentum_60 = close.pct_change(60)

    # volatility penalty
    volatility = close.pct_change().rolling(20).std()

    # combine components
    score = (
        0.5 * trend_strength +
        0.4 * momentum_60 -
        0.3 * volatility
    )

    return score.replace([np.inf, -np.inf], 0).fillna(0)
