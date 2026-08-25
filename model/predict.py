"""
Generate prediction scores for trading signals.
"""

import pandas as pd
import numpy as np


def generate_scores(model, price_data: dict) -> pd.DataFrame:
    """
    Generate prediction scores for all tickers in price_data.

    Args:
        model: Trained model instance
        price_data: Dict of ticker -> price DataFrame

    Returns:
        scores: DataFrame with ticker -> score mapping
    """
    scores = {}

    for ticker, price_df in price_data.items():
        if price_df.empty:
            print(f"[SKIP] No data for {ticker}")
            continue

        # Calculate features
        features = calculate_features(price_df)

        # Generate score
        if features is not None and not features.empty:
            score = compute_score(model, features)
            scores[ticker] = float(score) if score is not None else 50.0
        else:
            scores[ticker] = float(50.0)  # Default neutral score

    # Create DataFrame
    result = pd.DataFrame(list(scores.items()), columns=["ticker", "score"])
    result = result.sort_values("score", ascending=False)

    return result


def calculate_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features from price data.

    Args:
        price_df: Price DataFrame for a single ticker

    Returns:
        DataFrame with calculated features
    """
    df = price_df.copy()

    # Basic indicators
    df["Returns"] = df["Close"].pct_change().fillna(0).astype(float)
    df["Volatility"] = df["Returns"].rolling(window=20).std().fillna(0).astype(float)
    df["MA_5"] = df["Close"].rolling(window=5).mean().fillna(0).astype(float)
    df["MA_20"] = df["Close"].rolling(window=20).mean().fillna(0).astype(float)
    df["MA_50"] = df["Close"].rolling(window=50).mean().fillna(0).astype(float)
    df["RSI"] = calculate_rsi(df["Close"], window=14).fillna(50).astype(float)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df["Close"])
    df["MACD"] = df["MACD"].fillna(0).astype(float)
    df["MACD_Signal"] = df["MACD_Signal"].fillna(0).astype(float)
    df["MACD_Hist"] = df["MACD_Hist"].fillna(0).astype(float)

    return df


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(price_series: pd.Series) -> tuple:
    """Calculate MACD indicators."""
    exp1 = price_series.ewm(span=12, adjust=False).mean()
    exp2 = price_series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_score(model, features: pd.DataFrame) -> float:
    """
    Compute prediction score for a ticker.

    Args:
        model: Trained model (or None for default scoring)
        features: Feature DataFrame

    Returns:
        Score value
    """
    # Use last row of features
    last_features = features.iloc[-1]

    # Convert to array if needed
    if isinstance(last_features, pd.Series):
        feature_array = last_features.values.reshape(1, -1)
    else:
        feature_array = last_features

    # If model is not fitted, use default scoring based on features
    if not hasattr(model, "predict"):
        # Default scoring: use technical indicators
        score = _default_score(features)
        return float(score)

    # Predict using trained model
    try:
        pred = model.predict(feature_array)
        proba = model.predict_proba(feature_array)

        # Score based on probability
        score = proba[0][1] * 100 if len(proba[0]) > 1 else proba[0][0] * 100
    except Exception:
        # Fallback to default scoring if prediction fails
        score = _default_score(features)

    return float(score)


def _default_score(features: pd.DataFrame) -> float:
    """
    Default scoring based on technical indicators.

    Args:
        features: Feature DataFrame

    Returns:
        Score value (0-100)
    """
    score = 50.0  # Neutral score

    # RSI component
    if "RSI" in features.columns and "RSI" in features:
        rsi = features["RSI"].iloc[-1]
        if not pd.isna(rsi) and isinstance(rsi, (int, float)):
            if float(rsi) < 30:
                score += 20  # Undersold
            elif float(rsi) > 70:
                score -= 20  # Overbought

    # Momentum component
    if "Returns" in features.columns and "Returns" in features:
        ret = features["Returns"].iloc[-1]
        if not pd.isna(ret) and isinstance(ret, (int, float)):
            if float(ret) > 0.02:
                score += 15  # Strong gain
            elif float(ret) < -0.02:
                score -= 15  # Strong loss

    # Volatility penalty
    if "Volatility" in features.columns and "Volatility" in features:
        vol = features["Volatility"].iloc[-1]
        if not pd.isna(vol) and isinstance(vol, (int, float)):
            if float(vol) > 0.05:
                score -= 10  # High volatility penalty

    return float(max(0, min(100, score)))
