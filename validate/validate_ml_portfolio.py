from data_loader import load_price_data, load_nifty
from ml.predict import generate_ml_scores
from portfolio.portfolio_engine import run_portfolio_backtest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    price_data = load_price_data()
    nifty = load_nifty()

    # 🔁 THIS is the swap point
    scores = generate_ml_scores(price_data, nifty)

    results = run_portfolio_backtest(
        price_data=price_data,
        scores=scores,
        benchmark=nifty,
        equity_start=100000,
        top_n=10,
        cost_pct=0.001
    )

    print(results.tail())
    print("\nCAGR:", (results["Equity"].iloc[-1] / 100000) ** (252 / len(results)) - 1)
    print("Sharpe:", results["Daily Return"].mean() / results["Daily Return"].std() * np.sqrt(252))

if __name__ == "__main__":
    main()
