# portfolio/portfolio_engine.py
import pandas as pd
import numpy as np
from portfolio.risk_filters import market_exposure

EPS = 1e-9

def run_portfolio_backtest(
    price_data: dict,
    scores: pd.DataFrame,
    benchmark: pd.DataFrame,
    equity_start: float = 100000,
    top_n: int = 10,
    cost_pct: float = 0.001,
    rolling_window: int = 5,
    disp_median_window: int = 252
):
    """
    Multi-asset portfolio backtest with:
      - score smoothing (rolling average)
      - confidence-weighted exposure (Phase-4.3)
      - dynamic top-N selection
      - turnover-based transaction cost
      - position persistence
      - volatility weighting
      - soft risk-off exposure (market_exposure)
    
    Parameters
    ----------
    price_data : dict[ticker] -> DataFrame with Open,Close,Volume ...
    scores : pd.DataFrame
        index = dates, columns = tickers; ML scores (higher better)
    benchmark : pd.DataFrame or pd.Series
        benchmark price series or DataFrame (used by market_exposure)
    equity_start : float
    top_n : int
    cost_pct : float
        transaction cost per unit turnover (e.g., 0.001 = 0.1%)
    rolling_window : int
        smoothing window for scores
    disp_median_window : int
        window (days) to compute historical median dispersion
    """
    # --- smooth ML scores ---
    scores = scores.rolling(rolling_window, min_periods=1).mean()

    # --- precompute dispersion & historical median (cross-sectional) ---
    # dispersion per day: std across tickers
    dispersion = scores.std(axis=1)
    disp_med = dispersion.rolling(disp_median_window, min_periods=20).median()

    dates = scores.index
    equity = equity_start
    equity_curve = []
    daily_returns = []

    prev_weights = {}

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # --- market exposure baseline (soft risk-off) ---
        exposure_base = market_exposure(date, benchmark)  # 0.2/0.6/1.0 by default

        # --- confidence / dispersion logic ---
        disp_today = dispersion.get(date, np.nan)
        med = disp_med.get(date, np.nan)

        if pd.isna(disp_today) or pd.isna(med) or med <= EPS:
            conf_ratio = 1.0
        else:
            conf_ratio = float(disp_today / (med + EPS))

        # map conf_ratio to multiplier (tunable)
        # If conf_ratio ~ 1 => multiplier ~ 1.0
        # If conf_ratio << 1 => multiplier -> lower bound (weaker cross-sectional signal)
        # If conf_ratio >> 1 => multiplier -> slightly >1 (strong discrimination)
        conf_mult = 0.4 + 0.85 * conf_ratio  # linear mapping
        conf_mult = np.clip(conf_mult, 0.4, 1.25)

        # final exposure = baseline * confidence multiplier
        exposure = float(np.clip(exposure_base * conf_mult, 0.0, 1.5))

        # --- dynamic top_n based on confidence (conservative) ---
        # normalized factor in [0.6, 1.4] from conf_ratio
        dyn_factor = 0.6 + 0.8 * np.tanh(conf_ratio)  # smooth mapping
        cur_top_n = int(round(top_n * dyn_factor))
        cur_top_n = max(3, min(top_n, cur_top_n))

        # --- Rank universe using today's scores ---
        ranked = scores.loc[date].dropna().sort_values(ascending=False)
        selected = ranked.head(cur_top_n).index

        weights = {}
        total_weight = 0.0

        # --- Volatility-based sizing (uses prev_date vol) ---
        for ticker in selected:
            df = price_data.get(ticker)
            if df is None:
                continue
            if prev_date not in df.index or date not in df.index:
                continue

            returns = df["Open"].pct_change()
            vol = returns.rolling(20).std().loc[prev_date]

            if pd.isna(vol) or vol <= 0:
                continue

            w = 1.0 / vol
            weights[ticker] = w
            total_weight += w

        # --- Handle no positions (safety) ---
        if total_weight == 0:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            prev_weights = {}
            continue

        # --- Normalize weights so they sum to 1 ---
        for k in weights:
            weights[k] /= total_weight

        # --- Turnover calculation ---
        turnover = 0.0
        all_tickers = set(weights) | set(prev_weights)
        for t in all_tickers:
            turnover += abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0))

        prev_weights = weights.copy()

        # --- Compute portfolio return using open->open ---
        portfolio_ret = 0.0
        for ticker, w in weights.items():
            df = price_data[ticker]
            r = (df.loc[date, "Open"] / df.loc[prev_date, "Open"]) - 1.0
            portfolio_ret += w * r

        # --- Apply turnover cost and final exposure scaling ---
        portfolio_ret -= turnover * cost_pct
        portfolio_ret *= exposure

        equity *= (1.0 + portfolio_ret)
        equity_curve.append(equity)
        daily_returns.append(portfolio_ret)

    return pd.DataFrame({"Equity": equity_curve, "Daily Return": daily_returns}, index=dates[1:])
