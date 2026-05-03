import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio.portfolio_engine import run_portfolio_backtest


# ----------------------------
# METRICS
# ----------------------------
def compute_metrics(equity: pd.Series, daily_returns: pd.Series):
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }


# ----------------------------
# LOAD DATA (YOU ALREADY HAVE THIS)
# ----------------------------
def load_inputs():
    """
    Replace this with your actual loading logic.
    Must return:
      price_data: dict[ticker] -> DataFrame
      scores: DataFrame (index=dates, columns=tickers)
      benchmark: Series (NIFTY close)
    """
    from data_loader import load_price_data, load_scores, load_nifty

    price_data = load_price_data()
    scores = load_scores()
    benchmark = load_nifty()

    return price_data, scores, benchmark


# ----------------------------
# MAIN VALIDATION
# ----------------------------
def main():
    price_data, scores, nifty = load_inputs()

    print("\n=== RUNNING PHASE-2 PORTFOLIO BACKTEST ===")
    portfolio = run_portfolio_backtest(
        price_data=price_data,
        scores=scores,
        benchmark=nifty,
        top_n=10,
        cost_pct=0.001
    )

    equity = portfolio["Equity"]
    daily_returns = portfolio["Daily Return"]

    # Portfolio metrics
    port_metrics = compute_metrics(equity, daily_returns)

    # Benchmark metrics
    nifty_ret = nifty.pct_change().dropna()
    nifty_equity = (1 + nifty_ret).cumprod()

    bench_metrics = compute_metrics(nifty_equity, nifty_ret)

    print("\n--- PORTFOLIO METRICS ---")
    for k, v in port_metrics.items():
        print(f"{k:15s}: {v:.2%}")

    print("\n--- NIFTY METRICS ---")
    for k, v in bench_metrics.items():
        print(f"{k:15s}: {v:.2%}")

    # Plot equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(equity / equity.iloc[0], label="Portfolio")
    plt.plot(nifty_equity / nifty_equity.iloc[0], label="NIFTY")
    plt.legend()
    plt.title("Phase-2 Portfolio vs Benchmark")
    plt.grid()
    plt.show()

    # Rolling drawdown
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max

    plt.figure(figsize=(12, 4))
    plt.plot(dd, color="red")
    plt.title("Portfolio Drawdown")
    plt.grid()
    plt.show()

    print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()
