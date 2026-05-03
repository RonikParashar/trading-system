# data/live_data_loader.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

DEFAULT_LOOKBACK_DAYS = 400  # enough for rolling windows


def load_live_prices(
    tickers: List[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, pd.DataFrame]:
    """
    Phase-5.1 Live Data Loader

    Returns:
        price_data: dict[ticker -> DataFrame]
        DataFrame index: DatetimeIndex
        Required column: 'Open'
    """

    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    price_data = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                print(f"[WARN] Empty data for {ticker}")
                continue

            # Handle multi-index columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)

            # Standardize columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    print(f"[WARN] Missing {col} for {ticker}")
                    continue

            df = df[required_cols]

            # Standardize index
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            price_data[ticker] = df

        except Exception as e:
            print(f"[ERROR] Failed to load {ticker}: {e}")

    return price_data
