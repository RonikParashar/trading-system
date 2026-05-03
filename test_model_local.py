"""
Test ML Model on Local Dataset
Evaluates model performance with walk-forward cross-validation
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import ML modules
from ml.features import build_feature_matrix
from ml.labels import build_labels
from ml.model import train_model, predict
from ml.walkforward import walkforward_train_predict
from ml.metrics import information_coefficient, rank_ic

# Import portfolio modules
from portfolio.portfolio_engine import run_portfolio_backtest
from data_loader import load_price_data, load_nifty

import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading price data...")
print("=" * 60)

price_data = load_price_data()
nifty = load_nifty()

print(f"Loaded {len(price_data)} tickers: {list(price_data.keys())}")
print(f"NIFTY range: {nifty.index.min()} to {nifty.index.max()}")
print(f"Total rows per ticker: {[len(df) for df in price_data.values()]}")

# ============================================================
# 2. BUILD FEATURES & LABELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Building features and labels...")
print("=" * 60)

X = build_feature_matrix(price_data, nifty)
y = build_labels(price_data)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# ============================================================
# 3. ALIGN FEATURES AND LABELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Aligning features and labels...")
print("=" * 60)

# Align features and labels (both must have same index)
common_index = X.index.intersection(y.index)
X_aligned = X.loc[common_index]
y_aligned = y.loc[common_index]

print(f"Aligned shape: {X_aligned.shape}")
print(f"Aligned labels: {len(y_aligned)}")

# ============================================================
# 4. WALK-FORWARD ML PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Running walk-forward ML training...")
print("=" * 60)

preds = walkforward_train_predict(X_aligned, y_aligned, train_years=5, test_months=1)

print(f"Predictions generated: {len(preds)} rows")
print(f"Date range: {preds.index.get_level_values('Date').min()} to {preds.index.get_level_values('Date').max()}")

# ============================================================
# 5. MODEL EVALUATION METRICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Computing evaluation metrics...")
print("=" * 60)

# Align predictions with actual labels
idx = preds.index.intersection(y_aligned.index)
preds_aligned = preds.loc[idx]
y_final = y_aligned.loc[idx]

# Information Coefficient (Pearson correlation)
ic = information_coefficient(preds_aligned, y_final)

# Rank IC (Spearman correlation)
rank_ic_series = rank_ic(preds_aligned, y_final)
print(f"\nInformation Coefficient (IC):")
print(f"  Mean IC: {ic.mean():.4f}")
print(f"  Std IC: {ic.std():.4f}")
print(f"  IC t-stat: {ic.mean() / (ic.std() / np.sqrt(len(ic))):.2f}")
print(f"  Positive IC periods: {(ic > 0).sum()} / {len(ic)} ({(ic > 0).mean():.1%})")

print(f"\nRank IC:")
print(f"  Mean Rank IC: {rank_ic_series.mean():.4f}")
print(f"  Std Rank IC: {rank_ic_series.std():.4f}")

# Cumulative IC
rolling_ic = ic.rolling(252).mean()
print(f"\nLatest 1YT IC: {rolling_ic.iloc[-1]:.4f}" if not rolling_ic.iloc[-1] != rolling_ic.iloc[-1] else "N/A")

# ============================================================
# 5. CONVERT TO SCORES MATRIX
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Converting predictions to scores matrix...")
print("=" * 60)

scores = (
    preds
    .rename("score")
    .reset_index()
    .pivot(index="Date", columns="Ticker", values="score")
    .sort_index()
)

print(f"Scores matrix shape: {scores.shape}")
print(f"Date range: {scores.index.min()} to {scores.index.max()}")
print(f"Tickers: {list(scores.columns)}")

# ============================================================
# 6. PORTFOLIO BACKTEST
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Running portfolio backtest...")
print("=" * 60)

# Filter scores to valid date range
valid_dates = scores.index[scores.notna().any(axis=1)]
scores_filtered = scores.loc[valid_dates]

# Ensure price_data has matching dates
price_data_filtered = {}
for ticker, df in price_data.items():
    if scores.index.min() in df.index:
        start_idx = df.index.get_loc(scores.index.min())
        price_data_filtered[ticker] = df.iloc[start_idx:].copy()
    else:
        price_data_filtered[ticker] = df

portfolio = run_portfolio_backtest(
    price_data=price_data_filtered,
    scores=scores_filtered,
    benchmark=nifty,
    equity_start=100000,
    top_n=10,
    cost_pct=0.001,
)

print(f"Backtest complete: {len(portfolio)} days")

# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Computing performance metrics...")
print("=" * 60)

equity = portfolio["Equity"]
daily_returns = portfolio["Daily Return"]

# Portfolio metrics
cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
vol = daily_returns.std() * np.sqrt(252)
sharpe = cagr / vol if vol != 0 else 0
max_dd = (equity - equity.cummax()).min() / equity.cummax().max()

# Benchmark metrics
nifty_aligned = nifty.loc[scores_filtered.index]
nifty_ret = nifty_aligned.pct_change().dropna()
nifty_equity = (1 + nifty_ret).cumprod() * 100000

bench_cagr = (nifty_equity.iloc[-1] / nifty_equity.iloc[0]) ** (252 / len(nifty_equity)) - 1
bench_vol = nifty_ret.std() * np.sqrt(252)
bench_sharpe = bench_cagr / bench_vol if bench_vol != 0 else 0
bench_max_dd = (nifty_equity - nifty_equity.cummax()).min() / nifty_equity.cummax().max()

print("\n--- PORTFOLIO METRICS ---")
print(f"CAGR:           {cagr:>8.2%}")
print(f"Volatility:     {vol:>8.2%}")
print(f"Sharpe Ratio:   {sharpe:>8.2f}")
print(f"Max Drawdown:   {max_dd:>8.2%}")
print(f"Total Return:   {(equity.iloc[-1] / equity.iloc[0] - 1):>8.2%}")
print(f"Final Equity:   ${equity.iloc[-1]:,.2f}")

print("\n--- NIFTY BENCHMARK ---")
print(f"CAGR:           {bench_cagr:>8.2%}")
print(f"Volatility:     {bench_vol:>8.2%}")
print(f"Sharpe Ratio:   {bench_sharpe:>8.2f}")
print(f"Max Drawdown:   {bench_max_dd:>8.2%}")
print(f"Total Return:   {(nifty_equity.iloc[-1] / nifty_equity.iloc[0] - 1):>8.2%}")

print("\n--- EXCESS RETURNS ---")
print(f"Alpha (vs NIFTY): {(cagr - bench_cagr):>8.2%}")
print(f"Sharpe Diff:      {(sharpe - bench_sharpe):>8.2f}")

# ============================================================
# 8. PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Generating plots...")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Equity curves
ax1 = axes[0]
ax1.plot(equity / equity.iloc[0], label="Portfolio", linewidth=2)
ax1.plot(nifty_equity / nifty_equity.iloc[0], label="NIFTY", linewidth=2, alpha=0.7)
ax1.set_title("Portfolio vs NIFTY (Normalized)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Date")
ax1.set_ylabel("Normalized Value")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(1.0, color='black', linestyle='--', linewidth=0.5)

# Plot 2: Drawdown
ax2 = axes[1]
rolling_max = equity.cummax()
dd = (equity - rolling_max) / rolling_max
ax2.plot(dd, color='red', linewidth=2)
ax2.set_title("Portfolio Drawdown", fontsize=14, fontweight='bold')
ax2.set_xlabel("Date")
ax2.set_ylabel("Drawdown %")
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Plot 3: Rolling Sharpe (1Y)
ax3 = axes[2]
rolling_sharpe = (daily_returns.rolling(252).mean() * 252) / (daily_returns.rolling(252).std() * np.sqrt(252))
ax3.plot(rolling_sharpe, color='green', linewidth=2, label="Rolling 1Y Sharpe")
ax3.axhline(sharpe, color='blue', linestyle='--', label="Full Period Sharpe", alpha=0.7)
ax3.set_title("Rolling 1-Year Sharpe Ratio", fontsize=14, fontweight='bold')
ax3.set_xlabel("Date")
ax3.set_ylabel("Sharpe Ratio")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("test_results.png", dpi=150)
print("Plot saved to: test_results.png")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TEST COMPLETE - SUMMARY")
print("=" * 60)

# Determine model quality
quality = "GOOD" if (sharpe > 1.0 and ic.mean() > 0.02) else ("MODERATE" if sharpe > 0.5 else "NEEDS IMPROVEMENT")

print(f"""
Model Quality Assessment: {quality}

Key Findings:
  - IC Mean: {ic.mean():.4f} [{'PASS' if ic.mean() > 0.02 else 'WARN'}]
  - Rank IC Mean: {rank_ic_series.mean():.4f}
  - Portfolio Sharpe: {sharpe:.2f} [{'PASS' if sharpe > 1.0 else 'WARN'}]
  - Alpha vs NIFTY: {(cagr - bench_cagr):.2%} [{'PASS' if cagr > bench_cagr else 'WARN'}]
  - Max Drawdown: {max_dd:.2%} [{'PASS' if abs(max_dd) < 0.2 else 'WARN'}]

Next Steps:
  - If IC > 0.03: Model has predictive power
  - If Sharpe > 1.0: Risk-adjusted returns are good
  - If Alpha > 0: Outperforming benchmark
""")

plt.show()
