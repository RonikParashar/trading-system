"""
Daily Paper Trading Runner with ML Model
Uses saved model (ml/model.pkl) for live inference
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path

# Import modules
from data.live_data_loader import load_live_prices
from ml.features import build_feature_matrix
from data_loader import load_nifty
from paper_trading.paper_trade_runner import run_daily_paper_trade

# ============================================================
# CONFIGURATION
# ============================================================
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"
]

MODEL_PATH = Path("ml/model.pkl")
LOOKBACK_DAYS = 400  # enough for feature calculation

# ============================================================
# 1. LOAD MODEL
# ============================================================
print("=" * 60)
print("ML PAPER TRADING RUNNER")
print("=" * 60)

print(f"\nLoading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"Model type: {type(model).__name__}")
print(f"Features expected: {len(model.named_steps['scaler'].mean_)}")

# ============================================================
# 2. FETCH LIVE DATA
# ============================================================
print("\nFetching live price data...")
price_data = load_live_prices(tickers=TICKERS, lookback_days=LOOKBACK_DAYS)

valid_tickers = [t for t, df in price_data.items() if not df.empty and len(df) >= 200]
print(f"Valid tickers for inference: {len(valid_tickers)}")

if len(valid_tickers) < 5:
    print("[ERROR] Insufficient data for paper trading. Need at least 5 tickers with 200+ days.")
    exit(1)

# ============================================================
# 3. FETCH NIFTY BENCHMARK
# ============================================================
print("\nFetching NIFTY benchmark...")
benchmark = load_nifty(start="2020-01-01")
print(f"NIFTY data loaded: {len(benchmark)} rows")

# ============================================================
# 4. BUILD FEATURES & RUN INFERENCE
# ============================================================
print("\nBuilding features for ML inference...")

# Build features using same function as training
X = build_feature_matrix(price_data, benchmark)

# Flatten for prediction
features_flat = X.reset_index()

# Generate scores for the latest date
latest_date = features_flat["Date"].max()
# Convert to date string for consistency
latest_date_str = latest_date.strftime("%Y-%m-%d")
print(f"Latest date with features: {latest_date_str}")

latest_features = features_flat[features_flat["Date"] == latest_date]

# Predict for each ticker
scores_dict = {}
for _, row in latest_features.iterrows():
    ticker = row["Ticker"]
    feature_cols = [c for c in latest_features.columns if c not in ["Date", "Ticker"]]
    X_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols)

    try:
        score = model.predict(X_row)[0]
        scores_dict[ticker] = score
        print(f"  {ticker}: {score:.4f}")
    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")

# Create scores DataFrame
scores = pd.DataFrame(scores_dict, index=[pd.to_datetime(latest_date_str)])
scores = scores.sort_index()
print(f"\nScores shape: {scores.shape}")
print(f"Scores index: {scores.index}")

# ============================================================
# 5. RUN PAPER TRADE
# ============================================================
print("\n" + "=" * 60)
print("RUNNING PAPER TRADE")
print("=" * 60)

# Prepare price_data for the engine (need historical for backtest engine)
price_data_engine = {t: price_data[t] for t in scores_dict.keys()}

# Align benchmark
benchmark_aligned = benchmark.loc[:latest_date]

# Use string date for the runner
date_str = latest_date_str
print(f"Running paper trade for: {date_str}")

run_daily_paper_trade(
    date=date_str,
    price_data=price_data_engine,
    scores=scores,
    benchmark=benchmark_aligned,
)

print("\n" + "=" * 60)
print("PAPER TRADE COMPLETE")
print("=" * 60)

# Show current state
import json
with open("paper_trading/portfolio_state.json", "r") as f:
    state = json.load(f)

print(f"""
Current Portfolio State:
  Date: {state['date']}
  Equity: ${state['equity']:,.2f}
  Cash:   ${state['cash']:,.2f}
  Positions: {len(state['positions'])} stocks
""")

if state['positions']:
    print("\nHoldings:")
    for ticker, shares in state['positions'].items():
        if shares != 0:
            print(f"  {ticker}: {shares} shares")
