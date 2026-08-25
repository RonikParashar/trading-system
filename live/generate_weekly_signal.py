# live/generate_weekly_signal.py
"""
Weekly signal generation + paper trading update.

Runs once a week (via GitHub Actions, Saturday mornings IST):
  1. Pulls fresh live price data for the universe (config/universe.yaml)
  2. Computes the frozen factor strategy (momentum + low_vol + quality,
     top_n=20, same as validated in backtest)
  3. Marks the existing paper portfolio to market at current prices
  4. Rebalances into this week's target weights
  5. Logs everything to paper_trading/logs/ and updates portfolio_state.json

This is a paper-trading simulation only -- it never touches real money
or a real brokerage account.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from data.live_data_loader import load_live_prices
from factors.factor_scores import build_factor_matrix
from portfolio.risk_filters import market_exposure

CONFIG_PATH = "config/universe.yaml"
STATE_FILE = "paper_trading/portfolio_state.json"
LOG_DIR = "paper_trading/logs"
OUTPUT_DIR = "outputs"
COST_PCT = 0.001


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["universe"], cfg.get("top_n", 20)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {"equity": 100000, "cash": 100000, "positions": {}, "date": None}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def mark_to_market(positions, price_data, date):
    """Value existing positions at today's open."""
    value = 0.0
    for ticker, shares in positions.items():
        if ticker in price_data and date in price_data[ticker].index:
            value += shares * price_data[ticker].loc[date, "Open"]
        else:
            print(f"[WARN] No price for {ticker} on {date}, valuing at 0 for this mark")
    return value


def rebalance(equity, target_weights, price_data, date, cost_pct=COST_PCT):
    """Liquidate everything, buy into target weights. Returns (positions, cash)."""
    cash = equity  # already liquidated conceptually -- equity passed in is post-liquidation value
    positions = {}
    for ticker, weight in target_weights.items():
        if ticker not in price_data or date not in price_data[ticker].index:
            continue
        price = price_data[ticker].loc[date, "Open"]
        target_value = equity * weight
        shares = int(target_value / price)
        cost = shares * price * (1 + cost_pct)
        if shares > 0 and cost <= cash:
            positions[ticker] = shares
            cash -= cost
    return positions, cash


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers, top_n = load_config()
    print(f"Universe: {len(tickers)} tickers, top_n={top_n}")

    # -----------------------------
    # 1. Load live data
    # -----------------------------
    print("Fetching live price data...")
    price_data = load_live_prices(tickers, lookback_days=400)
    print(f"Loaded {len(price_data)} / {len(tickers)} tickers")

    if len(price_data) < 10:
        raise RuntimeError("Too few tickers loaded -- aborting rather than trading on bad data")

    nifty = yf.download("^NSEI", period="1y", progress=False, auto_adjust=True)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty_close = nifty["Close"]

    # -----------------------------
    # 2. Build factor scores (frozen strategy, no re-tuning here)
    # -----------------------------
    factors = build_factor_matrix(price_data)
    latest_date = factors.index.get_level_values("Date").max()
    today_scores = factors.xs(latest_date, level="Date")["composite"].dropna()

    ranked = today_scores.sort_values(ascending=False)
    selected = ranked.head(top_n)
    print(f"\nTop {len(selected)} picks for {latest_date.date()}:")
    print(selected)

    # -----------------------------
    # 3. Inverse-vol weighting + regime exposure (same as backtest)
    # -----------------------------
    target_weights = {}
    for ticker in selected.index:
        df = price_data[ticker]
        vol = df["Open"].pct_change().rolling(20).std().iloc[-1]
        if pd.isna(vol) or vol <= 0:
            continue
        target_weights[ticker] = 1.0 / vol

    total = sum(target_weights.values())
    target_weights = {t: w / total for t, w in target_weights.items()}

    exposure = market_exposure(latest_date, nifty_close)
    target_weights = {t: w * exposure for t, w in target_weights.items()}
    print(f"\nRegime exposure: {exposure:.2f}")

    # -----------------------------
    # 4. Save this week's signal (human-readable, for your weekend check-in)
    # -----------------------------
    signal_df = pd.DataFrame({
        "ticker": list(target_weights.keys()),
        "target_weight": list(target_weights.values()),
        "composite_score": [today_scores.get(t, float("nan")) for t in target_weights],
    }).sort_values("target_weight", ascending=False)

    date_str = latest_date.strftime("%Y-%m-%d")
    signal_path = f"{OUTPUT_DIR}/weekly_signal_{date_str}.csv"
    signal_df.to_csv(signal_path, index=False)
    print(f"\nSaved weekly signal to {signal_path}")

    # -----------------------------
    # 5. Update paper trading state
    # -----------------------------
    state = load_state()
    old_positions = state.get("positions", {})
    cash = state.get("cash", 100000)

    equity_before = cash + mark_to_market(old_positions, price_data, latest_date)
    print(f"\nMark-to-market equity before rebalance: {equity_before:,.2f}")

    new_positions, new_cash = rebalance(equity_before, target_weights, price_data, latest_date)
    equity_after = new_cash + mark_to_market(new_positions, price_data, latest_date)

    state.update({
        "equity": equity_after,
        "cash": new_cash,
        "positions": new_positions,
        "date": str(latest_date),
    })
    save_state(state)

    log = {
        "date": str(latest_date),
        "equity_before_rebalance": equity_before,
        "equity_after_rebalance": equity_after,
        "regime_exposure": exposure,
        "positions": new_positions,
        "target_weights": target_weights,
    }
    with open(f"{LOG_DIR}/{date_str}.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n[PAPER] {latest_date.date()} | Equity: {equity_after:,.2f} | "
          f"Positions: {len(new_positions)} | Exposure: {exposure:.2f}")


if __name__ == "__main__":
    main()