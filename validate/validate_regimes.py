import numpy as np
import pandas as pd

from data_loader import load_price_data, load_nifty
from ml.predict import generate_ml_scores
from portfolio.portfolio_engine import run_portfolio_backtest


def compute_regimes(nifty: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": nifty})

    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["Ret20"] = df["Close"].pct_change(20)
    df["Vol20"] = df["Close"].pct_change().rolling(20).std()

    vol_thresh = df["Vol20"].quantile(0.75)

    regimes = pd.DataFrame(index=df.index)
    regimes["Bull"] = df["Close"] > df["EMA200"]
    regimes["Bear"] = df["Close"] < df["EMA200"]
    regimes["HighVol"] = df["Vol20"] > vol_thresh
    regimes["Sideways"] = df["Ret20"].abs() < 0.03

    return regimes


def performance_metrics(returns: pd.Series):
    if returns.std() == 0:
        return np.nan, np.nan, np.nan

    cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_dd = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).sub(1).min()

    return cagr, sharpe, max_dd


def main():
    print("\n=== PHASE-4.1: MARKET REGIME STRESS TEST ===\n")

    price_data = load_price_data()
    nifty = load_nifty()

    scores = generate_ml_scores(price_data, nifty)

    results = run_portfolio_backtest(
        price_data=price_data,
        scores=scores,
        benchmark=nifty,
        equity_start=100000,
        top_n=10,
        cost_pct=0.001
    )

    returns = results["Daily Return"]
    regimes = compute_regimes(nifty).reindex(returns.index).dropna()

    report = []

    for regime in regimes.columns:
        mask = regimes[regime]

        if mask.sum() < 50:
            continue

        r = returns[mask]
        cagr, sharpe, max_dd = performance_metrics(r)

        report.append({
            "Regime": regime,
            "Days": mask.sum(),
            "CAGR": round(cagr * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Max Drawdown %": round(max_dd * 100, 2)
        })

    report_df = pd.DataFrame(report).set_index("Regime")
    print(report_df)


if __name__ == "__main__":
    main()
