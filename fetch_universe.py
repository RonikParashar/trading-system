# fetch_universe.py
"""
Downloads historical price data for the current Nifty 50 constituents
and saves them to data/processed/, in the same format the rest of the
pipeline already expects.

CAVEAT (read this): this uses TODAY's Nifty 50 list applied
retroactively back to 2015. That's a real improvement over the old
10 hand-picked mega-caps (50 independently-defined stocks vs. 10
the developer chose), but it's not a perfect point-in-time universe --
a few of these companies weren't in the index (or didn't exist in
their current form) for the whole backtest period. Those will just
have shorter history and get naturally excluded from any window that
needs more data than they have (the factor code already drops NaNs).
Good enough to test whether the earlier "alpha" survives outside the
original 10 -- not good enough to be the final word.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import time

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2010-01-01"

# Current Nifty 50 constituents (as of Dec 2025), NSE symbols
NIFTY_50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC",
    "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

failed = []
succeeded = []

for symbol in NIFTY_50_SYMBOLS:
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)

        if df is None or df.empty:
            print(f"[SKIP] {ticker}: no data returned")
            failed.append(symbol)
            continue

        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"

        out_path = PROC_DIR / f"{ticker}.csv"
        df.to_csv(out_path)
        print(f"[OK]   {ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
        succeeded.append(symbol)

        time.sleep(0.3)  # be polite to the API

    except Exception as e:
        print(f"[FAIL] {ticker}: {e}")
        failed.append(symbol)

print("\n" + "=" * 60)
print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
if failed:
    print(f"Failed symbols (check these manually): {failed}")