# data_loader.py
import pandas as pd
from pathlib import Path
import yfinance as yf

# import scoring + tradable functions you implemented earlier
from portfolio.scoring import compute_score
from portfolio.universe import tradable_universe

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

def load_price_data():
    """
    Load all processed CSVs from data/processed and return a dict:
      { 'TICKER': DataFrame(index=Date, columns=[Open,High,Low,Close,Volume,...]) }
    """
    price_data = {}
    for f in PROC_DIR.glob("*.csv"):
        df = pd.read_csv(f, parse_dates=["Date"], index_col="Date")
        # Ensure standard columns exist and are named as expected
        # (yfinance produced these in earlier steps)
        price_data[f.stem] = df.sort_index()
    return price_data


def load_scores():
    """
    Compute the scoring matrix (DataFrame) required by the portfolio engine.

    Returns:
      scores : pd.DataFrame
        index = union of dates across tickers (sorted),
        columns = tickers (e.g. RELIANCE.NS), values = score (float) or NaN if not tradable
    """
    price_data = load_price_data()
    scores_dict = {}

    for ticker, df in price_data.items():
        # compute score using your scoring.py
        try:
            score_ser = compute_score(df)
        except Exception as e:
            print(f"[WARN] compute_score failed for {ticker}: {e}")
            score_ser = pd.Series(dtype=float)

        # compute tradable mask, set non-tradable to NaN
        try:
            trad_mask = tradable_universe(df)
        except Exception as e:
            print(f"[WARN] tradable_universe failed for {ticker}: {e}")
            trad_mask = pd.Series(False, index=df.index)

        # align score index to df.index, and mask non-tradable
        score_ser = score_ser.reindex(df.index)
        score_ser[~trad_mask.reindex(df.index).fillna(False)] = pd.NA

        scores_dict[ticker] = score_ser

    # Build DataFrame (dates × tickers)
    scores_df = pd.DataFrame(scores_dict)
    # sort index
    scores_df = scores_df.sort_index()
    return scores_df


def load_nifty(ticker="^NSEI", start="2010-01-01"):
    """
    Download NIFTY index using yfinance (free) and return a pandas Series of Close prices.
    """
    import yfinance as yf
    import pandas as pd

    try:
        df = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=True
        )

        if df is None or df.empty:
            raise RuntimeError("Empty data returned for NIFTY")

        # Force Series explicitly
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        nifty = close.copy()
        nifty.name = "NIFTY"
        nifty = nifty.sort_index()

        return nifty

    except Exception as e:
        raise RuntimeError(f"Failed to download NIFTY index: {e}")
