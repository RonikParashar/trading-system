"""
Test ML Model on Live/Recent Data (Past 40 days)
Fetches fresh data from yfinance and evaluates model performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import ML modules
from ml.features import build_feature_matrix
from ml.inference import generate_features
from ml.labels import build_labels
from ml.model import train_model, predict, build_model
from ml.metrics import information_coefficient, rank_ic
from ml.inference import run_daily_ml_inference

# Import portfolio modules
from portfolio.portfolio_engine import run_portfolio_backtest
from data_loader import load_nifty

# Import live data loader
from data.live_data_loader import load_live_prices

import matplotlib.pyplot as plt
import joblib

# ============================================================
# 1. FETCH LIVE DATA (Past 40 days)
# ============================================================
print("=" * 60)
print("STEP 1: Fetching live data from yfinance...")
print("=" * 60)

import yfinance as yf

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS"
]

# Fetch 600 days for historical training
lookback = 600
end_date = datetime.today()
start_date = end_date - timedelta(days=lookback)

price_data = {}
for ticker in TICKERS:
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )

        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)  # Keep only 'Close', 'Open', etc.

        if df.empty or "Close" not in df.columns:
            print(f"[WARN] No data for {ticker}")
            continue

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        price_data[ticker] = df
        print(f"  {ticker}: {len(df)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to load {ticker}: {e}")

print(f"Loaded {len(price_data)} tickers")
for ticker, df in price_data.items():
    if not df.empty:
        print(f"  {ticker}: {len(df)} rows, {df.index.min()} to {df.index.max()}")

# Fetch NIFTY benchmark
hist_nifty = load_nifty(start=start_date.strftime("%Y-%m-%d"))

# Get recent 20 days subset
recent_price_data = {}
recent_start = end_date - timedelta(days=40)
for ticker, df in price_data.items():
    recent_price_data[ticker] = df[df.index >= recent_start]

# ============================================================
# 2. CHECK DATA QUALITY
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Checking data quality...")
print("=" * 60)

# Check for gaps and volume
for ticker, df in price_data.items():
    if df.empty:
        print(f"[WARN] {ticker}: Empty data")
        continue

    latest_date = df.index.max()
    latest_open = df["Open"].iloc[-1] if "Open" in df.columns else "N/A"
    latest_vol = df["Volume"].iloc[-1] if "Volume" in df.columns else "N/A"

    # Check if we have enough data for features
    if len(df) < 200:
        print(f"[WARN] {ticker}: Only {len(df)} days - may not have enough for 200d features")
    else:
        print(f"[OK] {ticker}: {len(df)} days, Latest Close: {latest_open:.2f}")

# ============================================================
# 3. TRAIN MODEL ON HISTORICAL DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training model on historical data...")
print("=" * 60)

model_path = Path("ml/model.pkl")

print(f"Historical data: {[len(df) for df in price_data.values()]} rows per ticker")

# Build features and labels for training
X_hist = build_feature_matrix(price_data, hist_nifty)
y_hist = build_labels(price_data)

# Align
common_idx = X_hist.index.intersection(y_hist.index)
X_hist = X_hist.loc[common_idx]
y_hist = y_hist.loc[common_idx]

print(f"Training data: {X_hist.shape[0]} samples")

# Train final model
model = train_model(X_hist, y_hist)
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# ============================================================
# 4. RUN DAILY INFERENCE ON RECENT DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Running daily inference on recent 20 days...")
print("=" * 60)

# Get the last 20 trading days from recent data
recent_dates = []
for ticker, df in price_data.items():
    if not df.empty:
        dates = df.index[-20:].tolist()
        recent_dates.extend(dates)

if recent_dates:
    recent_dates = sorted(set(recent_dates))[-20:]
    print(f"Recent date range: {recent_dates[0]} to {recent_dates[-1]}")

# Run inference for each recent day using the same features as training
daily_predictions = {}

# Build features for ALL tickers at once (to match training structure)
print("Building features for all tickers...")
all_features = build_feature_matrix(price_data, hist_nifty)
print(f"Features shape: {all_features.shape}")
print(f"Index names: {all_features.index.names}")

# Convert to DataFrame format matching training
# Reset index to get Date and Ticker as columns
features_flat = all_features.reset_index()
print(f"Columns: {features_flat.columns.tolist()}")

for ticker in TICKERS:
    ticker_feats = features_flat[features_flat["Ticker"] == ticker].copy()
    if ticker_feats.empty:
        print(f"[SKIP] {ticker}: No features")
        continue

    ticker_feats = ticker_feats.sort_values("Date")

    # Get predictions for last 20 days
    ticker_preds = {}
    for i in range(-20, 0):
        if abs(i) > len(ticker_feats):
            continue
        row = ticker_feats.iloc[i]
        date = row["Date"]

        # Build feature vector (exclude Date, Ticker columns)
        feature_cols = [c for c in ticker_feats.columns if c not in ["Date", "Ticker"]]
        X_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols)

        try:
            score = model.predict(X_row)[0]
            ticker_preds[date] = score
        except Exception as e:
            pass

    daily_predictions[ticker] = ticker_preds
    print(f"[OK] {ticker}: {len(ticker_preds)} predictions generated")

# Convert to scores DataFrame (date x ticker)
scores_dict = {}
for ticker, preds in daily_predictions.items():
    for date, score in preds.items():
        if date not in scores_dict:
            scores_dict[date] = {}
        scores_dict[date][ticker] = score

scores_df = pd.DataFrame(scores_dict)
scores_df = scores_df.sort_index()

# Fix: if columns are dates instead of tickers, transpose
if len(scores_df.columns) > 10:
    scores_df = scores_df.T  # Now dates are index, tickers are columns

print(f"\nScores generated: {scores_df.shape}")
print(f"Date range: {scores_df.index.min()} to {scores_df.index.max()}")
print(f"Tickers: {list(scores_df.columns)}")

# ============================================================
# 5. COMPUTE ACTUAL RETURNS (What the model predicted)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Computing actual 20-day forward returns...")
print("=" * 60)

# For each prediction date, check what the actual return was
actual_returns = {}
horizon = 20  # 20-day forward return

for ticker, df in price_data.items():
    if df.empty:
        continue

    df = df.sort_index()

    # Compute forward returns
    fwd_ret = df["Close"].pct_change(horizon).shift(-horizon)

    for date in scores_df.index:
        if date in fwd_ret.index and not pd.isna(fwd_ret.loc[date]):
            if ticker not in actual_returns:
                actual_returns[ticker] = {}
            actual_returns[ticker][date] = fwd_ret.loc[date]

# ============================================================
# 6. EVALUATE PREDICTION ACCURACY
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Evaluating prediction accuracy...")
print("=" * 60)

# Build prediction vs actual DataFrame
eval_data = []
for date in scores_df.index:
    for ticker in scores_df.columns:
        try:
            pred = scores_df.loc[date, ticker]
            if pd.isna(pred):
                continue
            actual = actual_returns.get(ticker, {}).get(date)
            if actual is not None and not pd.isna(actual):
                eval_data.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Prediction": pred,
                    "Actual": actual
                })
        except:
            continue

eval_df = pd.DataFrame(eval_data)

if len(eval_df) > 0:
    # Overall correlation
    corr = eval_df["Prediction"].corr(eval_df["Actual"])
    print(f"Overall Prediction vs Actual Correlation: {corr:.4f}")

    # Rank correlation
    rank_corr = eval_df["Prediction"].rank().corr(eval_df["Actual"].rank())
    print(f"Rank Correlation (Spearman): {rank_corr:.4f}")

    # Directional accuracy (did high predictions have high actual returns?)
    # Split into quintiles
    eval_df["Pred_Quintile"] = pd.qcut(eval_df["Prediction"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

    print("\nAverage Actual Return by Prediction Quintile:")
    quintile_stats = eval_df.groupby("Pred_Quintile")["Actual"].agg(["mean", "std", "count"])
    print(quintile_stats)

    # Hit rate (positive actual when positive prediction)
    pos_pred = eval_df[eval_df["Prediction"] > 0]
    pos_actual_rate = (pos_pred["Actual"] > 0).mean()
    print(f"\nHit rate when prediction > 0: {pos_actual_rate:.1%}")

    neg_pred = eval_df[eval_df["Prediction"] < 0]
    neg_actual_rate = (neg_pred["Actual"] > 0).mean()
    print(f"Hit rate when prediction < 0: {neg_actual_rate:.1%}")

else:
    print("[WARN] No evaluation data - forward returns may not be available yet")

# ============================================================
# 7. RECENT MARKET VOLATILITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Analyzing recent market volatility...")
print("=" * 60)

# Compute rolling volatility for each ticker
vol_analysis = {}
for ticker, df in price_data.items():
    if df.empty or len(df) < 20:
        continue

    ret = df["Close"].pct_change()
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()

    recent_vol = vol_20.iloc[-1] if not pd.isna(vol_20.iloc[-1]) else vol_20.iloc[-5]
    hist_vol = vol_20.mean()

    vol_analysis[ticker] = {
        "Recent Vol (20d)": recent_vol,
        "Historic Vol": hist_vol,
        "Vol Ratio": recent_vol / hist_vol if hist_vol > 0 else 1.0
    }

print("\nVolatility Comparison (Recent vs Historical):")
for ticker, stats in vol_analysis.items():
    ratio = stats["Vol Ratio"]
    status = "ELEVATED" if ratio > 1.2 else ("NORMAL" if ratio > 0.8 else "LOW")
    print(f"  {ticker}: {ratio:.2f}x [{status}]")

# ============================================================
# 8. PREDICTION DISTRIBUTION
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Prediction distribution analysis...")
print("=" * 60)

print("\nScore statistics across all predictions:")
print(f"  Mean:   {np.mean(scores_df.values):.4f}")
print(f"  Std:    {np.std(scores_df.values):.4f}")
print(f"  Min:    {np.min(scores_df.values):.4f}")
print(f"  Max:    {np.max(scores_df.values):.4f}")
print(f"  Median: {np.median(scores_df.values):.4f}")

# Per-ticker average scores
print("\nAverage score by ticker:")
for ticker in scores_df.columns:
    avg_score = scores_df[ticker].mean()
    print(f"  {ticker}: {avg_score:.4f}")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("LIVE TEST COMPLETE - SUMMARY")
print("=" * 60)

# Market regime assessment
avg_vol_ratio = np.mean([s["Vol Ratio"] for s in vol_analysis.values()])
market_state = "HIGHLY VOLATILE" if avg_vol_ratio > 1.5 else ("VOLATILE" if avg_vol_ratio > 1.2 else "NORMAL")

print(f"""
Market Condition: {market_state}
  Average Volatility Ratio: {avg_vol_ratio:.2f}x

Model Performance on Recent Data:
  - Predictions generated: {len(scores_df)} dates x {len(scores_df.columns)} tickers
  - Score range: [{scores_df.values.min():.4f}, {scores_df.values.max():.4f}]
  - Score std: {scores_df.values.std():.4f} (cross-sectional dispersion)
""")

if len(eval_df) > 0:
    print(f"""
Prediction Accuracy (where forward returns available):
  - Correlation: {corr:.4f}
  - Rank IC: {rank_corr:.4f}
  - Hit rate (positive pred): {pos_actual_rate:.1%}
""")

print("""
Notes:
  - Recent 20 days showed elevated volatility due to geopolitical events
  - Model predictions are based on technical features (momentum, trend, vol)
  - Forward returns may not be fully available for most recent dates
  - Higher score dispersion indicates more trading opportunities
""")

# ============================================================
# 10. PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Generating plots...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Score heatmap
ax1 = axes[0, 0]
if len(scores_df) > 1:
    im = ax1.imshow(scores_df.values.T, aspect='auto', cmap='RdYlGn')
    ax1.set_yticks(range(len(scores_df.columns)))
    ax1.set_yticklabels(scores_df.columns)
    ax1.set_xlabel("Date")
    ax1.set_title("ML Scores Heatmap (Green=High, Red=Low)")
    plt.colorbar(im, ax=ax1, label="Score")

# Plot 2: Score distribution
ax2 = axes[0, 1]
ax2.hist(scores_df.values.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(scores_df.values.mean(), color='red', linestyle='--', label=f"Mean: {scores_df.values.mean():.3f}")
ax2.set_xlabel("Score")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of ML Scores")
ax2.legend()

# Plot 3: Volatility comparison
ax3 = axes[1, 0]
tickers = list(vol_analysis.keys())
vol_ratios = [vol_analysis[t]["Vol Ratio"] for t in tickers]
ax3.barh(tickers, vol_ratios, color='steelblue')
ax3.axvline(1.0, color='red', linestyle='--', label="Historical Avg")
ax3.set_xlabel("Volatility Ratio (Recent/Historical)")
ax3.set_title("Recent Volatility vs Historical")
ax3.legend()

# Plot 4: Scores over time (aggregate)
ax4 = axes[1, 1]
scores_mean = scores_df.mean(axis=1)
scores_std = scores_df.std(axis=1)
ax4.plot(scores_mean.index, scores_mean.values, label="Mean Score", linewidth=2)
ax4.fill_between(scores_mean.index,
                  scores_mean.values - scores_std.values,
                  scores_mean.values + scores_std.values,
                  alpha=0.3, label="1 Std Dev")
ax4.set_title("Aggregate ML Score Over Time")
ax4.set_xlabel("Date")
ax4.set_ylabel("Score")
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("test_results_live.png", dpi=150)
print("Plot saved to: test_results_live.png")

plt.show()
