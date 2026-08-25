# factors/factor_scores.py
"""
Transparent factor scoring -- replaces the ML model (ridge regression) with
three well-documented, auditable equity factors. Every score is causal
(only ever uses data up to and including the row's own date).

Factors:
  - momentum : 12-month return, skipping the most recent month
               (the "12-1" convention -- avoids the well-documented
               short-term reversal effect)
  - low_vol  : negative of 60-day realized volatility (lower vol = higher score)
  - quality  : shallowness of drawdown over the trailing year, as a price-only
               proxy for business stability. NOTE: this is NOT a fundamentals-based
               quality factor (no ROE/debt data available) -- it's the closest
               honest approximation from price data alone. Treat it as weaker
               evidence than momentum/low_vol.
"""

import numpy as np
import pandas as pd


def compute_momentum(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """12-1 month momentum: return from (t-lookback) to (t-skip)."""
    return close.shift(skip) / close.shift(lookback) - 1


def compute_low_vol(close: pd.Series, window: int = 60) -> pd.Series:
    """Negative realized volatility -- higher score = steadier stock."""
    vol = close.pct_change().rolling(window).std()
    return -vol


def compute_quality_proxy(close: pd.Series, window: int = 252) -> pd.Series:
    """Shallowness of trailing drawdown -- closer to 0 (less negative) = 'higher quality'."""
    roll_max = close.rolling(window).max()
    drawdown = close / roll_max - 1
    return drawdown


def build_factor_matrix(price_data: dict) -> pd.DataFrame:
    """
    Builds a (Date, Ticker)-indexed factor matrix with raw factors,
    per-date cross-sectional z-scores, and a composite score.
    """
    frames = {}

    for ticker, df in price_data.items():
        df = df.sort_index()
        close = df["Close"]

        feat = pd.DataFrame({
            "momentum": compute_momentum(close),
            "low_vol": compute_low_vol(close),
            "quality": compute_quality_proxy(close),
        })

        feat = feat.dropna()
        feat["Ticker"] = ticker
        feat = feat.set_index("Ticker", append=True)
        frames[ticker] = feat

    if not frames:
        raise RuntimeError("No factor data built for any ticker")

    factors = pd.concat(frames.values()).sort_index()
    factors.index = factors.index.set_names(["Date", "Ticker"])

    # --- Cross-sectional z-scores (per date, across the universe) ---
    for col in ["momentum", "low_vol", "quality"]:
        grp = factors.groupby(level="Date")[col]
        mean = grp.transform("mean")
        std = grp.transform("std")
        factors[f"{col}_z"] = (factors[col] - mean) / std.replace(0, np.nan)

    # --- Composite score: equal-weight average of the three factors ---
    factors["composite"] = factors[
        ["momentum_z", "low_vol_z", "quality_z"]
    ].mean(axis=1)

    return factors


def to_weekly_signal(factors: pd.DataFrame, rebalance_day: str = "W-FRI") -> pd.DataFrame:
    """
    Converts a daily composite score into a weekly-refreshed signal:
    the score only updates on each week's rebalance day, and is
    forward-filled through the week otherwise. This mirrors a real
    weekly rebalance workflow (check in once a week, not every day),
    and cuts unnecessary turnover/costs versus a daily-refreshed signal.

    Also applies a 2-trading-day safety shift: a score computed from
    data through date D is only used for decisions from D+2 onward,
    so a trade never relies on same-day or next-day data it couldn't
    have actually seen yet.
    """
    scores = (
        factors["composite"]
        .reset_index()
        .pivot(index="Date", columns="Ticker", values="composite")
        .sort_index()
    )

    # Only keep the value on each week's last trading day, forward-fill the rest
    is_rebalance_day = scores.index.isin(
        scores.resample(rebalance_day).last().index.intersection(scores.index)
    )
    weekly = scores.where(pd.Series(is_rebalance_day, index=scores.index)).ffill()

    # Safety shift: don't let the engine act on a score before it could exist
    weekly = weekly.shift(2)

    return weekly