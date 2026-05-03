import yfinance as yf
import yaml
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

with open("config/universe.yaml") as f:
    symbols = yaml.safe_load(f)["universe"]

for symbol in symbols:
    print(f"Downloading {symbol}")

    df = yf.download(
        symbol,
        start="2010-01-01",
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    # FIX: Flatten columns if multi-index
    if isinstance(df.columns, tuple) or hasattr(df.columns, "levels"):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Standardize column order
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.to_csv(DATA_DIR / f"{symbol}.csv")

print("Download complete.")
