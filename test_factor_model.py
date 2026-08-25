"""
Test Factor-Based Strategy (replaces ML model)
Weekly-refreshed, transparent momentum + low-vol + quality composite,
run through the same portfolio engine used for the ML version so results
are directly comparable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from factors.factor_scores import build_factor_matrix, to_weekly_signal
from portfolio.portfolio_engine import run_portfolio_backtest
from data_loader import load_price_data, load_nifty

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading price data...")
print("=" * 60)

price_data = load_price_data()
nifty = load_nifty()

print(f"Loaded {len(price_data)} tickers: {list(price_data.keys())}")

# ============================================================
# 2. BUILD FACTOR SCORES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Building factor scores...")
print("=" * 60)

factors = build_factor_matrix(price_data)
print(f"Factor matrix shape: {factors.shape}")
print(f"Factors: momentum, low_vol, quality (+ z-scores, + composite)")

weekly_scores = to_weekly_signal(factors)
print(f"Weekly signal shape: {weekly_scores.shape}")
print(f"Date range: {weekly_scores.index.min()} to {weekly_scores.index.max()}")

# ============================================================
# 3. PORTFOLIO BACKTEST (same engine as the ML version)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Running portfolio backtest...")
print("=" * 60)

valid_dates = weekly_scores.index[weekly_scores.notna().any(axis=1)]
scores_filtered = weekly_scores.loc[valid_dates]

price_data_filtered = {}
for ticker, df in price_data.items():
    if scores_filtered.index.min() in df.index:
        start_idx = df.index.get_loc(scores_filtered.index.min())
        price_data_filtered[ticker] = df.iloc[start_idx:].copy()
    else:
        price_data_filtered[ticker] = df

portfolio = run_portfolio_backtest(
    price_data=price_data_filtered,
    scores=scores_filtered,
    benchmark=nifty,
    equity_start=100000,
    top_n=20,
    cost_pct=0.001,
    rolling_window=1,  # weekly signal is already smoothed by design
)

print(f"Backtest complete: {len(portfolio)} days")

# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Computing performance metrics...")
print("=" * 60)

equity = portfolio["Equity"]
daily_returns = portfolio["Daily Return"]

cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
vol = daily_returns.std() * np.sqrt(252)
sharpe = cagr / vol if vol != 0 else 0
max_dd = (equity - equity.cummax()).min() / equity.cummax().max()

nifty_aligned = nifty.loc[scores_filtered.index.intersection(nifty.index)]
nifty_ret = nifty_aligned.pct_change().dropna()
nifty_equity = (1 + nifty_ret).cumprod() * 100000

bench_cagr = (nifty_equity.iloc[-1] / nifty_equity.iloc[0]) ** (252 / len(nifty_equity)) - 1
bench_vol = nifty_ret.std() * np.sqrt(252)
bench_sharpe = bench_cagr / bench_vol if bench_vol != 0 else 0
bench_max_dd = (nifty_equity - nifty_equity.cummax()).min() / nifty_equity.cummax().max()

print("\n--- PORTFOLIO METRICS (Factor Strategy) ---")
print(f"CAGR:           {cagr:>8.2%}")
print(f"Volatility:     {vol:>8.2%}")
print(f"Sharpe Ratio:   {sharpe:>8.2f}")
print(f"Max Drawdown:   {max_dd:>8.2%}")
print(f"Total Return:   {(equity.iloc[-1] / equity.iloc[0] - 1):>8.2%}")

print("\n--- NIFTY BENCHMARK ---")
print(f"CAGR:           {bench_cagr:>8.2%}")
print(f"Volatility:     {bench_vol:>8.2%}")
print(f"Sharpe Ratio:   {bench_sharpe:>8.2f}")
print(f"Max Drawdown:   {bench_max_dd:>8.2%}")

print("\n--- EXCESS RETURNS ---")
print(f"Alpha (vs NIFTY): {(cagr - bench_cagr):>8.2%}")
print(f"Sharpe Diff:      {(sharpe - bench_sharpe):>8.2f}")

# ============================================================
# 5. PLOTS
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

ax1 = axes[0]
ax1.plot(equity / equity.iloc[0], label="Factor Portfolio", linewidth=2)
ax1.plot(nifty_equity / nifty_equity.iloc[0], label="NIFTY", linewidth=2, alpha=0.7)
ax1.set_title("Factor Strategy vs NIFTY (Normalized)", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(1.0, color='black', linestyle='--', linewidth=0.5)

ax2 = axes[1]
rolling_max = equity.cummax()
dd = (equity - rolling_max) / rolling_max
ax2.plot(dd, color='red', linewidth=2)
ax2.set_title("Factor Strategy Drawdown", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig("test_results_factor.png", dpi=150)
print("\nPlot saved to: test_results_factor.png")