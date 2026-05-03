import random
import numpy as np
import pandas as pd

from data_loader import load_price_data, load_nifty
from ml.predict import generate_ml_scores
from portfolio.portfolio_engine import run_portfolio_backtest


def universe_subset(price_data, drop_frac, seed):
    """
    Randomly remove a fraction of the universe
    """
    random.seed(seed)
    tickers = list(price_data.keys())
    n_drop = int(len(tickers) * drop_frac)
    drop = set(random.sample(tickers, n_drop))

    return {k: v for k, v in price_data.items() if k not in drop}


def compute_metrics(results):
    returns = results["Daily Return"]
    equity = results["Equity"]

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_dd = ((equity / equity.cummax()) - 1).min()

    return round(cagr * 100, 2), round(sharpe, 2), round(max_dd * 100, 2)


def main():
    print("\n=== PHASE-5.3: UNIVERSE ROBUSTNESS TEST ===\n")

    base_price_data = load_price_data()
    nifty = load_nifty()

    scores = generate_ml_scores(base_price_data, nifty)

    drop_fracs = [0.0, 0.2, 0.4, 0.6]
    seeds = [42, 99, 123]

    records = []

    for frac in drop_fracs:
        for seed in seeds:
            price_data = universe_subset(base_price_data, frac, seed)

            # Align scores to surviving universe
            valid_tickers = price_data.keys()
            sub_scores = scores[valid_tickers]

            results = run_portfolio_backtest(
                price_data=price_data,
                scores=sub_scores,
                benchmark=nifty,
                top_n=10,
                rolling_window=5,
                cost_pct=0.001
            )

            cagr, sharpe, maxdd = compute_metrics(results)

            records.append({
                "Dropped %": int(frac * 100),
                "Seed": seed,
                "CAGR": cagr,
                "Sharpe": sharpe,
                "MaxDD": maxdd
            })

    df = pd.DataFrame(records)

    print("---- Individual Runs ----")
    print(df)

    print("\n---- Aggregate (Mean by Drop %) ----")
    print(
        df.groupby("Dropped %")[["CAGR", "Sharpe", "MaxDD"]]
        .mean()
        .round(2)
    )


if __name__ == "__main__":
    main()
