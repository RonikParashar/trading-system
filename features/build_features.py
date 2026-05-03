import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(exist_ok=True)

for file in RAW_DIR.glob("*.csv"):
    df = pd.read_csv(file, parse_dates=["Date"], index_col="Date")

    # Basic features
    df["ret_1d"] = df["Close"].pct_change()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["vol_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    df.dropna(inplace=True)
    df.to_csv(OUT_DIR / file.name)
