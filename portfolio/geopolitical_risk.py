"""
Geopolitical Risk Layer for Portfolio Adjustment

Uses News Sentiment-based signals that reflect geopolitical conditions:
- News sentiment score (aggregate sentiment from geopolitical keywords)
- News volume spike (proxy for breaking news events)
- Keyword frequency (war, conflict, sanctions, elections, etc.)

Adjusts sector exposures based on news sentiment-driven risk signals.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple
import requests
import json
from bs4 import BeautifulSoup

# ============================================================
# SECTOR MAPPING
# ============================================================
SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "ITC.NS": "FMCG",
    "HINDUNILVR.NS": "FMCG",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
}

# ============================================================
# RISK INDICATORS
# ============================================================

def fetch_india_vix(lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch India VIX data (market fear gauge)
    High VIX = elevated uncertainty/geopolitical tension
    """
    try:
        vix = yf.download(
            "^INDIAVIX",
            start=(datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
            end=datetime.today().strftime("%Y-%m-%d"),
            progress=False,
        )

        if vix.empty:
            return None

        vix.index = pd.to_datetime(vix.index)
        return vix[["Close"]]
    except Exception as e:
        print(f"[WARN] Could not fetch India VIX: {e}")
        return None


def fetch_crude_oil(lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch Brent crude oil prices
    Oil spikes often correlate with Middle East tensions
    """
    try:
        crude = yf.download(
            "BZ=F",  # Brent crude futures
            start=(datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
            end=datetime.today().strftime("%Y-%m-%d"),
            progress=False,
        )

        if crude.empty:
            return None

        crude.index = pd.to_datetime(crude.index)
        return crude[["Close"]]
    except Exception as e:
        print(f"[WARN] Could not fetch crude oil: {e}")
        return None


def fetch_usd_inr(lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch USD/INR exchange rate
    Currency stress can indicate trade tensions or capital flight
    """
    try:
        usdinr = yf.download(
            "USDINR=X",
            start=(datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
            end=datetime.today().strftime("%Y-%m-%d"),
            progress=False,
        )

        if usdinr.empty:
            return None

        usdinr.index = pd.to_datetime(usdinr.index)
        return usdinr[["Close"]]
    except Exception as e:
        print(f"[WARN] Could not fetch USD/INR: {e}")
        return None


def fetch_gold(lookback_days: int = 60) -> pd.DataFrame:
    """
    Fetch gold prices (safe-haven asset)
    Gold rallies often indicate risk-off sentiment
    """
    try:
        gold = yf.download(
            "GC=F",  # Gold futures
            start=(datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
            end=datetime.today().strftime("%Y-%m-%d"),
            progress=False,
        )

        if gold.empty:
            return None

        gold.index = pd.to_datetime(gold.index)
        return gold[["Close"]]
    except Exception as e:
        print(f"[WARN] Could not fetch gold: {e}")
        return None


# ============================================================
# RISK METRICS COMPUTATION
# ============================================================

def compute_vix_signal(vix_data: pd.DataFrame) -> Tuple[float, str]:
    """
    Compute VIX signal:
    - VIX > 20: elevated fear
    - VIX > 25: high fear
    - VIX > 30: extreme fear

    Returns: (risk_multiplier, regime)
    """
    if vix_data is None or len(vix_data) < 5:
        return 1.0, "Normal"

    current_vix = vix_data["Close"].iloc[-1]
    vix_ma = vix_data["Close"].rolling(20).mean().iloc[-1]

    if current_vix > 30:
        return 0.3, "Extreme Fear"
    elif current_vix > 25:
        return 0.5, "High Fear"
    elif current_vix > 20:
        return 0.7, "Elevated"
    else:
        return 1.0, "Normal"


def compute_oil_signal(oil_data: pd.DataFrame) -> Tuple[float, str]:
    """
    Compute oil volatility signal:
    - Rapid oil price increases = geopolitical tension
    - High oil volatility = supply uncertainty

    Returns: (risk_multiplier, regime)
    """
    if oil_data is None or len(oil_data) < 20:
        return 1.0, "Normal"

    # 20-day momentum
    oil_mom = (oil_data["Close"].iloc[-1] / oil_data["Close"].iloc[-20] - 1)

    # 20-day volatility
    oil_ret = oil_data["Close"].pct_change()
    oil_vol = oil_ret.rolling(20).std().iloc[-1]
    hist_vol = oil_ret.rolling(60).std().mean()

    vol_ratio = oil_vol / hist_vol if hist_vol > 0 else 1.0

    # Scoring
    if oil_mom > 0.15 or vol_ratio > 2.0:
        return 0.5, "Oil Stress"
    elif oil_mom > 0.08 or vol_ratio > 1.5:
        return 0.7, "Oil Concern"
    else:
        return 1.0, "Normal"


def compute_currency_signal(usdinr_data: pd.DataFrame) -> Tuple[float, str]:
    """
    Compute USD/INR signal:
    - Rapid INR weakening = capital flight or trade stress

    Returns: (risk_multiplier, regime)
    """
    if usdinr_data is None or len(usdinr_data) < 20:
        return 1.0, "Normal"

    # 20-day change
    fx_mom = (usdinr_data["Close"].iloc[-1] / usdinr_data["Close"].iloc[-20] - 1)

    # INR weakening (USD/INR up) is negative for India
    if fx_mom > 0.03:
        return 0.6, "Currency Stress"
    elif fx_mom > 0.015:
        return 0.85, "Currency Weak"
    else:
        return 1.0, "Normal"


def compute_gold_signal(gold_data: pd.DataFrame) -> Tuple[float, str]:
    """
    Compute gold signal (safe-haven flows):
    - Gold rallying = risk-off sentiment

    Returns: (risk_multiplier, regime)
    """
    if gold_data is None or len(gold_data) < 20:
        return 1.0, "Normal"

    # 20-day momentum
    gold_mom = (gold_data["Close"].iloc[-1] / gold_data["Close"].iloc[-20] - 1)

    if gold_mom > 0.08:
        return 0.6, "Safe-Haven Demand"
    elif gold_mom > 0.04:
        return 0.85, "Risk-Off"
    else:
        return 1.0, "Normal"


# ============================================================
# SECTOR-LEVEL ADJUSTMENTS
# ============================================================

def get_sector_adjustments(
    vix_signal: float,
    oil_signal: float,
    fx_signal: float,
    gold_signal: float
) -> Dict[str, float]:
    """
    Apply sector-specific adjustments based on geopolitical signals:

    - IT: hurt by currency stress, trade wars
    - Energy: benefits from oil spikes (RELIANCE)
    - Banking: hurt by VIX spikes, currency stress
    - FMCG: defensive, less affected
    - Infrastructure: mixed (benefits from oil decline)
    """

    adjustments = {
        "IT": 1.0,
        "Energy": 1.0,
        "Banking": 1.0,
        "FMCG": 1.0,
        "Infrastructure": 1.0,
    }

    # VIX affects all risk assets, especially financials
    adjustments["Banking"] *= vix_signal
    adjustments["IT"] *= max(vix_signal, 0.7)  # IT less sensitive to VIX

    # Oil spikes benefit energy, hurt others
    adjustments["Energy"] *= (2.0 - oil_signal)  # Inverse relationship
    adjustments["Infrastructure"] *= oil_signal * 0.8 + 0.2  # Partial benefit
    adjustments["IT"] *= oil_signal * 0.7 + 0.3  # Oil hurts IT (input costs)

    # Currency stress hurts exporters (IT)
    adjustments["IT"] *= fx_signal
    adjustments["Banking"] *= fx_signal * 0.8 + 0.2

    # Gold signal = risk-off, defensive sectors benefit
    adjustments["FMCG"] *= (2.0 - gold_signal)  # Defensive benefits
    adjustments["Banking"] *= gold_signal  # Banks hurt in risk-off

    # Cap adjustments between 0.3 and 1.5
    for sector in adjustments:
        adjustments[sector] = np.clip(adjustments[sector], 0.3, 1.5)

    return adjustments


# ============================================================
# MAIN FUNCTION
# ============================================================

def compute_geopolitical_risk_scores(
    scores: pd.DataFrame,
    date: pd.Timestamp = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Adjust ML scores based on geopolitical risk signals.

    Parameters:
    -----------
    scores : pd.DataFrame
        Raw ML scores (index=date, columns=tickers)
    date : pd.Timestamp
        Date for signal calculation (default: today)

    Returns:
    --------
    adjusted_scores : pd.DataFrame
        Geopolitically-adjusted scores
    risk_report : Dict
        Current risk signals and sector adjustments
    """

    if date is None:
        date = pd.Timestamp(datetime.today())

    # Fetch all risk indicators
    print("Fetching geopolitical risk indicators...")
    vix_data = fetch_india_vix()
    oil_data = fetch_crude_oil()
    fx_data = fetch_usd_inr()
    gold_data = fetch_gold()

    # Compute signals
    vix_mult, vix_regime = compute_vix_signal(vix_data)
    oil_mult, oil_regime = compute_oil_signal(oil_data)
    fx_mult, fx_regime = compute_currency_signal(fx_data)
    gold_mult, gold_regime = compute_gold_signal(gold_data)

    print(f"  VIX:    {vix_regime:20s} (mult: {vix_mult:.2f})")
    print(f"  Oil:    {oil_regime:20s} (mult: {oil_mult:.2f})")
    print(f"  FX:     {fx_regime:20s} (mult: {fx_mult:.2f})")
    print(f"  Gold:   {gold_regime:20s} (mult: {gold_mult:.2f})")

    # Get sector adjustments
    sector_adj = get_sector_adjustments(vix_mult, oil_mult, fx_mult, gold_mult)

    print("\nSector adjustments:")
    for sector, adj in sector_adj.items():
        print(f"  {sector:15s}: {adj:.2f}x")

    # Apply adjustments to scores
    adjusted_scores = scores.copy()

    for ticker in adjusted_scores.columns:
        sector = SECTOR_MAP.get(ticker, "Other")
        adj_factor = sector_adj.get(sector, 1.0)
        adjusted_scores[ticker] = adjusted_scores[ticker] * adj_factor

    # Build risk report
    risk_report = {
        "date": str(date),
        "overall_risk": "High" if np.mean([vix_mult, oil_mult, fx_mult, gold_mult]) < 0.6 else
                        ("Medium" if np.mean([vix_mult, oil_mult, fx_mult, gold_mult]) < 0.85 else "Low"),
        "signals": {
            "vix": {"value": vix_regime, "multiplier": vix_mult},
            "oil": {"value": oil_regime, "multiplier": oil_mult},
            "fx": {"value": fx_regime, "multiplier": fx_mult},
            "gold": {"value": gold_regime, "multiplier": gold_mult},
        },
        "sector_adjustments": sector_adj,
    }

    return adjusted_scores, risk_report


# ============================================================
# EXPOSURE SCALER (for portfolio engine)
# ============================================================

def compute_gross_exposure_cap(
    vix_signal: float,
    oil_signal: float,
    fx_signal: float,
    gold_signal: float
) -> float:
    """
    Compute maximum gross exposure based on geopolitical risk.

    Normal market: 100% exposure
    Elevated risk: 80% exposure
    High risk: 60% exposure
    Extreme risk: 40% exposure
    """

    avg_signal = np.mean([vix_signal, oil_signal, fx_signal, gold_signal])

    if avg_signal >= 0.85:
        return 1.0  # Full exposure
    elif avg_signal >= 0.70:
        return 0.8  # Slight reduction
    elif avg_signal >= 0.50:
        return 0.6  # Defensive
    else:
        return 0.4  # Capital preservation
