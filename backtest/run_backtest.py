import vectorbt as vbt
import pandas as pd
from pathlib import Path
from strategies.baseline_trend import generate_signals

DATA_DIR = Path("data/processed")

portfolios = []

for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file, parse_dates=["Date"], index_col="Date")

    entries, exits = generate_signals(df)

    pf = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        fees=0.001,          # 0.1% cost
        slippage=0.001,      # realistic slippage
        init_cash=100000,
        freq="1D"
    )

    print(f"\n=== {file.stem} ===")
    print(pf.stats())
    portfolios.append(pf)
