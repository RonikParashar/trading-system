import pandas as pd

def tradable_universe(df: pd.DataFrame,
                      min_avg_volume: int = 500_000,
                      max_gap: float = 0.20) -> pd.Series:
    """
    Returns a boolean Series indicating if the stock
    is tradable on each date.
    """

    # basic validity
    valid_price = df["Close"] > 0

    # liquidity filter
    avg_volume = df["Volume"].rolling(20).mean()
    liquid = avg_volume >= min_avg_volume

    # gap filter
    prev_close = df["Close"].shift(1)
    gap = (df["Open"] - prev_close).abs() / prev_close
    no_extreme_gap = gap < max_gap

    tradable = valid_price & liquid & no_extreme_gap
    return tradable.fillna(False)
