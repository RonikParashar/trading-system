import numpy as np
import pandas as pd

from data_loader import load_price_data, load_nifty
from ml.predict import generate_ml_scores
from portfolio.portfolio_engine import run_portfolio_backtest


def compute_metrics(results, equity_start=100000):
    equity = results["Equity"]
    daily_ret = results["Daily Return"]

    cagr = (equity.iloc[-1] / equity_start) ** (252 / len(equity)) - 1
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    drawdown = (equity / equity.cummax() - 1).min()

    return {
        "CAGR": cagr * 100,
        "Sharpe": sharpe,
        "MaxDD": drawdown * 100
    }


def main():
    print("\n=== PHASE-5.2: PARAMETER SENSITIVITY TEST ===\n")

    price_data = load_price_data()
    benchmark = load_nifty()

    scores = generate_ml_scores(price_data, benchmark)

    # -----------------------
    # Parameter grids
    # -----------------------
    top_n_list = [5, 8, 10, 12, 15]
    rolling_windows = [3, 5, 7, 10]
    cost_pcts = [0.0005, 0.001, 0.002]

    records = []

    for top_n in top_n_list:
        for rw in rolling_windows:
            for cost in cost_pcts:
                results = run_portfolio_backtest(
                    price_data=price_data,
                    scores=scores,
                    benchmark=benchmark,
                    top_n=top_n,
                    rolling_window=rw,
                    cost_pct=cost
                )

                metrics = compute_metrics(results)

                records.append({
                    "top_n": top_n,
                    "rolling_window": rw,
                    "cost_pct": cost,
                    **metrics
                })

    df = pd.DataFrame(records)

    # -----------------------
    # Summary views
    # -----------------------
    print("---- Aggregate Performance ----")
    print(df.groupby("top_n")[["CAGR", "Sharpe", "MaxDD"]].mean().round(2))
    print()

    print("---- Rolling Window Sensitivity ----")
    print(df.groupby("rolling_window")[["CAGR", "Sharpe", "MaxDD"]].mean().round(2))
    print()

    print("---- Transaction Cost Sensitivity ----")
    print(df.groupby("cost_pct")[["CAGR", "Sharpe", "MaxDD"]].mean().round(2))
    print()

    # -----------------------
    # Failure detection
    # -----------------------
    failures = df[
        (df["CAGR"] < 0) |
        (df["Sharpe"] < 0.5) |
        (df["MaxDD"] < -25)
    ]

    print("---- ⚠️ Parameter Failure Cases ----")
    if failures.empty:
        print("✅ No catastrophic parameter failures detected.")
    else:
        print(failures.round(2))


if __name__ == "__main__":
    main()
