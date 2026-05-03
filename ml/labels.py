import pandas as pd
import numpy as np


def build_labels(
    price_data: dict,
    horizon: int = 20
) -> pd.Series:
    """
    Builds forward return labels for ML.

    Output:
      MultiIndex Series:
        index = (date, ticker)
        value = forward return (%)
    """
    rows = []

    for ticker, df in price_data.items():
        if "Close" not in df.columns:
            continue

        df = df.sort_index()

        # Forward return
        fwd_ret = (
            df["Close"]
            .pct_change(horizon)
            .shift(-horizon)
        )

        # Convert to list of dicts for MultiIndex DataFrame
        for date, score in fwd_ret.dropna().items():
            rows.append({
                'Date': date,
                'Ticker': ticker,
                'score': score
            })

    if not rows:
        return pd.Series([], dtype=float, name='score')

    # Create DataFrame and set MultiIndex
    df_labels = pd.DataFrame(rows)
    df_labels = df_labels.set_index(['Date', 'Ticker'])
    df_labels.index = df_labels.index.set_names(["Date", "Ticker"])

    return df_labels['score'].sort_index()
