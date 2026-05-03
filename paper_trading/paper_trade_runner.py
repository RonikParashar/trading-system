import json
import os
from datetime import datetime

import pandas as pd
import numpy as np

from portfolio.portfolio_engine import run_portfolio_backtest
from portfolio.risk_filters import market_exposure

STATE_FILE = "paper_trading/portfolio_state.json"
LOG_DIR = "paper_trading/logs"

# -----------------------------
# RISK LIMITS (PHASE-4)
# -----------------------------
MAX_POSITION_PCT = 0.10      # 10% per stock
MAX_GROSS_EXPOSURE = 1.00    # 100% invested
MAX_DAILY_TURNOVER = 0.25   # 25% of equity per day

# Phase-6.2 defaults (same as backtest)
EQUITY_START = 100000
TOP_N = 10
COST_PCT = 0.001
ROLLING_WINDOW = 5
DISP_MEDIAN_WINDOW = 252


# -----------------------------
# Utilities
# -----------------------------
def load_state():
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


# -----------------------------
# Main Runner - Phase-6.2 Engine
# -----------------------------
def run_daily_paper_trade(
    date,
    price_data,
    scores,
    benchmark,
    equity_start=None,
):
    # Ensure date is a Timestamp for consistent indexing
    if isinstance(date, str):
        date = pd.to_datetime(date)

    ensure_log_dir()
    state = load_state()

    # Use passed equity_start or fall back to state equity
    equity_prev = equity_start if equity_start is not None else state["equity"]
    cash = state["cash"]
    positions = state["positions"].copy()

    # -----------------------------
    # Build benchmark up to today
    # -----------------------------
    if isinstance(benchmark, pd.DataFrame):
        benchmark_today = benchmark.loc[:date]
    else:
        benchmark_today = benchmark

    # -----------------------------
    # Phase-6.2 Backtest Engine
    # -----------------------------
    # Run the full portfolio engine for all dates up to today
    # We'll capture today's positions and equity from the engine
    try:
        backtest_results = run_portfolio_backtest(
            price_data=price_data,
            scores=scores,
            benchmark=benchmark_today,
            equity_start=equity_prev,
            top_n=TOP_N,
            cost_pct=COST_PCT,
            rolling_window=ROLLING_WINDOW,
            disp_median_window=DISP_MEDIAN_WINDOW,
        )

        # Get the last equity from backtest
        if len(backtest_results) > 0:
            equity = backtest_results["Equity"].iloc[-1]
        else:
            equity = equity_prev

    except Exception as e:
        # Fallback to simple runner if Phase-6.2 fails
        print(f"[WARN] Phase-6.2 engine failed: {e}")
        print("Falling back to simplified scoring path...")
        _run_fallback_paper_trade(
            date=date,
            price_data=price_data,
            scores=scores,
            benchmark=benchmark_today,
            equity_prev=equity_prev,
            positions=positions,
            cash=cash,
        )
        state = load_state()
        return

    # -----------------------------
    # Reconstruct positions from backtest state
    # -----------------------------
    # Since Phase-6.2 returns equity curve, we need to track weights
    # We'll simulate the engine internally to get positions
    _reconstruct_positions_from_engine(
        price_data=price_data,
        scores=scores,
        benchmark=benchmark_today,
        date=date,
        equity=equity,
        positions=positions,
        cash=cash,
    )

    # -----------------------------
    # Save state
    # -----------------------------
    state.update({
        "equity": equity,
        "cash": cash,
        "positions": positions,
        "date": str(date),
    })
    save_state(state)

    # -----------------------------
    # Daily log
    # -----------------------------
    log = {
        "date": str(date),
        "equity": equity,
        "cash": cash,
        "positions": positions,
        "gross_exposure": sum(
            abs(positions[t] * price_data[t].loc[date, "Open"])
            for t in positions
        ) / equity if equity > 0 else 0,
    }

    # Use date string without colons for filename
    date_str = str(date).replace(":", "-").replace(" ", "_")
    with open(f"{LOG_DIR}/{date_str}.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"[PAPER] {date} | Equity: {equity:,.2f} | Positions: {len(positions)}")


# -----------------------------
# Fallback Runner (simplified)
# -----------------------------
def _run_fallback_paper_trade(
    date,
    price_data,
    scores,
    benchmark,
    equity_prev,
    positions,
    cash,
):
    # Ensure date is a Timestamp for consistent indexing
    if isinstance(date, str):
        date = pd.to_datetime(date)

    # Get today's ML scores
    today_scores = scores.loc[date].dropna()

    if today_scores.empty:
        print(f"[PAPER] No valid scores for {date}")
        return

    # Convert scores to weights
    # Use softmax-like normalization: higher score = higher weight
    # Only take positive scores for long positions
    positive_scores = today_scores[today_scores > 0]

    if positive_scores.empty:
        print(f"[PAPER] No positive scores for {date} - staying in cash")
        # Update state with no changes
        state = load_state()
        state["date"] = str(date)
        save_state(state)
        return

    # Normalize to weights (sum to 1)
    score_sum = positive_scores.sum()
    raw_weights = {t: s / score_sum for t, s in positive_scores.items()}

    # Target equity (simple: assume we capture some alpha from scores)
    # For paper trading, we use actual portfolio returns
    target_equity = equity_prev  # will be adjusted by holdings

    # Risk control: cap position size
    weights = {}
    for k, w in raw_weights.items():
        weights[k] = min(w, MAX_POSITION_PCT)

    # Risk control: cap total exposure
    gross = sum(abs(w) for w in weights.values())
    if gross > MAX_GROSS_EXPOSURE:
        scale = MAX_GROSS_EXPOSURE / gross
        weights = {k: w * scale for k, w in weights.items()}

    # Convert weights -> shares
    trades = {}
    turnover_value = 0.0

    for ticker, weight in weights.items():
        price = price_data[ticker].loc[date, "Open"]

        target_value = target_equity * weight
        target_shares = int(target_value / price)

        prev_shares = positions.get(ticker, 0)
        delta = target_shares - prev_shares

        trade_value = abs(delta) * price
        turnover_value += trade_value

        # Turnover safety
        if turnover_value > MAX_DAILY_TURNOVER * equity_prev:
            continue

        # Cash safety
        if delta > 0 and cash < delta * price:
            delta = int(cash / price)

        if delta != 0:
            trades[ticker] = delta
            cash -= delta * price
            positions[ticker] = prev_shares + delta

    # Remove empty positions
    positions = {k: v for k, v in positions.items() if v != 0}

    # Final equity
    holdings_value = sum(
        positions[t] * price_data[t].loc[date, "Open"]
        for t in positions
    )

    equity = cash + holdings_value

    # Save state
    state = load_state()
    state.update({
        "equity": equity,
        "cash": cash,
        "positions": positions,
        "date": str(date),
    })
    save_state(state)

    # Daily log
    log = {
        "date": str(date),
        "equity": equity,
        "cash": cash,
        "positions": positions,
        "trades": trades,
        "gross_exposure": sum(
            abs(positions[t] * price_data[t].loc[date, "Open"])
            for t in positions
        ) / equity if equity > 0 else 0,
    }

    # Use date string without colons for filename
    date_str = str(date).replace(":", "-").replace(" ", "_")
    with open(f"{LOG_DIR}/{date_str}.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"[PAPER] {date} | Equity: {equity:,.2f} | Trades: {len(trades)}")


# -----------------------------
# Reconstruct positions from engine state
# -----------------------------
def _reconstruct_positions_from_engine(
    price_data,
    scores,
    benchmark,
    date,
    equity,
    positions,
    cash,
):
    """
    Reconstruct portfolio positions by running the engine's scoring logic.
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    # Smooth ML scores (same as Phase-6.2)
    scores_smooth = scores.rolling(ROLLING_WINDOW, min_periods=1).mean()
    today_scores = scores_smooth.loc[date]

    # Dispersion calculation
    dispersion = scores_smooth.std(axis=1)
    disp_today = dispersion.get(date, np.nan)
    disp_med = dispersion.rolling(DISP_MEDIAN_WINDOW, min_periods=20).median().get(date, np.nan)

    # Confidence logic
    if pd.isna(disp_today) or pd.isna(disp_med) or disp_med <= 1e-9:
        conf_ratio = 1.0
    else:
        conf_ratio = float(disp_today / (disp_med + 1e-9))

    # Regime-aware exposure
    exposure_base = market_exposure(date, benchmark)

    # Confidence multiplier
    conf_mult = 0.4 + 0.85 * conf_ratio
    conf_mult = np.clip(conf_mult, 0.4, 1.25)

    # Exposure
    exposure = float(np.clip(exposure_base * conf_mult, 0.0, 1.5))

    # Dynamic Top-N
    dyn_factor = 0.6 + 0.8 * np.tanh(conf_ratio)
    cur_top_n = int(round(TOP_N * dyn_factor))
    cur_top_n = max(3, min(TOP_N, cur_top_n))

    # Select top tickers - only positive scores after smoothing
    pos_scores = today_scores[today_scores > 0].dropna()
    selected = pos_scores.sort_values(ascending=False).head(cur_top_n).index.tolist()
    print(f"[PAPER] Phase-6.2: Top {len(selected)} tickers selected for weights (positive scores only)")

    # Calculate volatility-based target weights
    target_weights = {}
    for ticker in selected:
        df = price_data.get(ticker)
        if df is None:
            continue

        returns = df["Open"].pct_change()
        vol = returns.rolling(20).std().loc[date]

        if pd.isna(vol) or vol <= 0:
            continue

        w = 1.0 / vol
        target_weights[ticker] = w

    if not target_weights:
        print("[PAPER] No valid tickers for volatility sizing")
        return positions

    # Normalize target weights
    total_weight = sum(target_weights.values())
    if total_weight <= 0:
        print("[PAPER] No valid weights computed")
        return positions

    for k in target_weights:
        target_weights[k] /= total_weight

    # Apply exposure scaling
    for k in target_weights:
        target_weights[k] *= exposure

    # Convert to positions using target weights
    total_target_value = sum(target_weights[t] * equity for t in target_weights)
    if total_target_value <= 0:
        print("[PAPER] Invalid target value")
        return positions

    # Normalize again after exposure
    scale = equity / total_target_value
    for ticker, weight in target_weights.items():
        target_value = weight * equity
        price = price_data[ticker].loc[date, "Open"]

        # Calculate target shares (rounded to whole shares)
        target_shares = int(target_value / price)

        # Ensure we have valid positions
        if target_shares > 0:
            positions[ticker] = target_shares

    return positions
