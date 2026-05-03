# portfolio/portfolio_engine.py

import pandas as pd
import numpy as np

from portfolio.risk_filters import market_exposure
from portfolio.slippage import estimate_slippage

EPS = 1e-9


def run_portfolio_backtest(
    price_data: dict,
    scores: pd.DataFrame,
    benchmark: pd.DataFrame,
    equity_start: float = 100000,
    top_n: int = 10,
    cost_pct: float = 0.001,
    rolling_window: int = 5,
    disp_median_window: int = 252,
):
    """
    Phase-6.2 Portfolio Engine:
      - Score smoothing
      - Regime-aware exposure
      - Confidence-weighted exposure
      - Dynamic top-N
      - Volatility sizing
      - Turnover-aware weight smoothing
      - Confidence-gated trading
      - Correct slippage + cost handling
    """

    # --------------------------------------------------
    # Smooth ML scores
    # --------------------------------------------------
    scores = scores.rolling(rolling_window, min_periods=1).mean()

    # --------------------------------------------------
    # Dispersion (cross-sectional confidence)
    # --------------------------------------------------
    dispersion = scores.std(axis=1)
    disp_med = dispersion.rolling(disp_median_window, min_periods=20).median()

    dates = scores.index

    equity = equity_start
    equity_curve = []
    daily_returns = []

    prev_weights = {}

    # --------------------------------------------------
    # Main Backtest Loop
    # --------------------------------------------------
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # ============================
        # Regime-aware base exposure
        # ============================
        exposure_base = market_exposure(date, benchmark)  # 0.2 / 0.6 / 1.0

        # ============================
        # Confidence logic
        # ============================
        disp_today = dispersion.get(date, np.nan)
        med = disp_med.get(date, np.nan)

        if pd.isna(disp_today) or pd.isna(med) or med <= EPS:
            conf_ratio = 1.0
        else:
            conf_ratio = float(disp_today / (med + EPS))

        # Confidence multiplier
        conf_mult = 0.4 + 0.85 * conf_ratio
        conf_mult = np.clip(conf_mult, 0.4, 1.25)

        exposure = float(np.clip(exposure_base * conf_mult, 0.0, 1.5))

        # ============================
        # Regime-conditioned rebalance speed (α)
        # ============================
        if exposure_base >= 0.9:        # Bull
            alpha = 0.40
        elif exposure_base >= 0.5:      # Neutral
            alpha = 0.25
        else:                           # Bear
            alpha = 0.10

        # Further dampening if confidence is low
        alpha *= np.clip(conf_ratio, 0.5, 1.2)

        # ============================
        # Dynamic Top-N
        # ============================
        dyn_factor = 0.6 + 0.8 * np.tanh(conf_ratio)
        cur_top_n = int(round(top_n * dyn_factor))
        cur_top_n = max(3, min(top_n, cur_top_n))

        ranked = scores.loc[date].dropna().sort_values(ascending=False)
        selected = ranked.head(cur_top_n).index

        # ============================
        # Target weights (volatility-based)
        # ============================
        target_weights = {}
        total_weight = 0.0

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
            target_weights[ticker] = w
            total_weight += w

        if total_weight <= 0:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            prev_weights = {}
            continue

        for k in target_weights:
            target_weights[k] /= total_weight

        # ============================
        # Phase-6.2: Turnover-Aware Weight Blending
        # ============================
        blended_weights = {}

        all_tickers = set(target_weights) | set(prev_weights)

        # Confidence-based trade gating
        min_trade_threshold = 0.02 / max(conf_ratio, 0.5)

        for t in all_tickers:
            w_prev = prev_weights.get(t, 0.0)
            w_target = target_weights.get(t, 0.0)

            if abs(w_target - w_prev) < min_trade_threshold:
                w_new = w_prev
            else:
                w_new = (1 - alpha) * w_prev + alpha * w_target

            if w_new > 0:
                blended_weights[t] = w_new

        # Normalize blended weights
        total_blended = sum(blended_weights.values())
        if total_blended <= 0:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            prev_weights = {}
            continue

        for k in blended_weights:
            blended_weights[k] /= total_blended

        # ============================
        # Turnover & Slippage (FIXED)
        # ============================
        turnover = 0.0
        slippage_cost = 0.0
        portfolio_ret = 0.0

        for t in set(blended_weights) | set(prev_weights):
            w_new = blended_weights.get(t, 0.0)
            w_old = prev_weights.get(t, 0.0)
            trade_fraction = abs(w_new - w_old)
            turnover += trade_fraction

            if trade_fraction > 0 and t in price_data:
                returns = price_data[t]["Open"].pct_change().loc[:prev_date]
                slippage_cost += estimate_slippage(
                    returns=returns,
                    trade_fraction=trade_fraction
                )

        # ============================
        # Portfolio Return (Open→Open)
        # ============================
        for t, w in blended_weights.items():
            df = price_data[t]
            raw_ret = (df.loc[date, "Open"] / df.loc[prev_date, "Open"]) - 1
            portfolio_ret += w * raw_ret

        # Apply costs
        portfolio_ret -= slippage_cost
        portfolio_ret -= turnover * cost_pct
        portfolio_ret *= exposure

        equity *= (1.0 + portfolio_ret)

        equity_curve.append(equity)
        daily_returns.append(portfolio_ret)

        prev_weights = blended_weights.copy()

    return pd.DataFrame(
        {"Equity": equity_curve, "Daily Return": daily_returns},
        index=dates[1:]
    )