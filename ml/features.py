# ml/features.py

import numpy as np
import pandas as pd


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def trend_slope(series, window=20):
    """
    Rolling linear regression slope (normalized)
    """
    def _slope(y):
        x = np.arange(len(y))
        if y.isna().any():
            return np.nan
        return np.polyfit(x, y, 1)[0] / y.mean()

    return series.rolling(window).apply(_slope, raw=False)


def rolling_beta(stock_ret, market_ret, window=60):
    cov = stock_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    return cov / var


def build_features(
    df: pd.DataFrame,
    market_close: pd.Series
) -> pd.DataFrame:
    """
    Build ML features for a single stock.
    """
    close = df["Close"]
    volume = df["Volume"]

    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    ret_60 = close.pct_change(60)

    vol_20 = close.pct_change().rolling(20).std()
    vol_60 = close.pct_change().rolling(60).std()

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    drawdown_60 = close / close.rolling(60).max() - 1

    market_ret = market_close.pct_change()
    rel_ret_20 = ret_20 - market_ret.reindex(close.index)
    rel_ret_60 = ret_60 - market_ret.reindex(close.index)

    beta_60 = rolling_beta(close.pct_change(), market_ret)

    vol_z = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    features = pd.DataFrame({
        "ret_5": ret_5,
        "ret_20": ret_20,
        "ret_60": ret_60,

        "vol_20": vol_20,
        "vol_60": vol_60,

        "ema20_ratio": close / ema20 - 1,
        "ema50_ratio": close / ema50 - 1,
        "ema200_ratio": close / ema200 - 1,

        "trend_20": trend_slope(close, 20),

        "drawdown_60": drawdown_60,

        "rel_ret_20": rel_ret_20,
        "rel_ret_60": rel_ret_60,

        "beta_60": beta_60,

        "vol_z": vol_z,
    })

    return features


def add_cross_sectional_ranks(features_dict):
    """
    Add cross-sectional percentile ranks per date.
    Output: MultiIndex columns (ticker, feature)
    """
    stacked = pd.concat(features_dict, axis=1)

    rank_cols = [
        "ret_20",
        "ret_60",
        "rel_ret_20",
        "rel_ret_60",
        "vol_20",
    ]

    tickers = stacked.columns.levels[0]

    for col in rank_cols:
        if col not in stacked.columns.levels[1]:
            continue

        values = stacked.xs(col, level=1, axis=1)
        ranks = values.rank(axis=1, pct=True)

        for ticker in tickers:
            stacked[(ticker, f"{col}_rank")] = ranks[ticker]

    return stacked.sort_index(axis=1)


def build_feature_matrix(price_data: dict, nifty: pd.Series) -> pd.DataFrame:
    """
    Builds ML feature matrix with MultiIndex (Date, Ticker)
    Returns:
        DataFrame indexed by (Date, Ticker)
    """

    features_dict = {}

    for ticker, df in price_data.items():
        try:
            df = df.copy().sort_index()

            # --- BASIC RETURNS ---
            df["ret_5"] = df["Close"].pct_change(5)
            df["ret_20"] = df["Close"].pct_change(20)
            df["ret_60"] = df["Close"].pct_change(60)

            # --- TREND FEATURES ---
            df["ema20"] = df["Close"].ewm(span=20).mean()
            df["ema50"] = df["Close"].ewm(span=50).mean()
            df["ema200"] = df["Close"].ewm(span=200).mean()

            df["ema20_ratio"] = df["Close"] / df["ema20"] - 1
            df["ema50_ratio"] = df["Close"] / df["ema50"] - 1
            df["ema200_ratio"] = df["Close"] / df["ema200"] - 1

            df["trend_20"] = df["ema20"] / df["ema50"] - 1

            # --- VOLATILITY ---
            df["vol_20"] = df["ret_5"].rolling(20).std()
            df["vol_60"] = df["ret_5"].rolling(60).std()

            df["vol_z"] = (
                df["vol_20"] - df["vol_20"].rolling(60).mean()
            ) / df["vol_20"].rolling(60).std()

            # --- DRAWDOWN ---
            roll_max = df["Close"].rolling(60).max()
            df["drawdown_60"] = df["Close"] / roll_max - 1

            # --- BETA TO NIFTY ---
            aligned = pd.concat(
                [
                    df["ret_20"],
                    nifty.pct_change(20)
                ],
                axis=1,
                join="inner"
            ).dropna()

            if not aligned.empty:
                cov = aligned.iloc[:, 0].rolling(60).cov(aligned.iloc[:, 1])
                var = aligned.iloc[:, 1].rolling(60).var()
                df.loc[aligned.index, "beta_60"] = cov / var
            else:
                df["beta_60"] = np.nan

            # --- RELATIVE RETURNS ---
            df["rel_ret_20"] = df["ret_20"] - nifty.pct_change(20)
            df["rel_ret_60"] = df["ret_60"] - nifty.pct_change(60)

            # --- SELECT FINAL FEATURES ---
            feature_cols = [
                "ret_5", "ret_20", "ret_60",
                "ema20_ratio", "ema50_ratio", "ema200_ratio",
                "trend_20",
                "vol_20", "vol_60", "vol_z",
                "drawdown_60",
                "beta_60",
                "rel_ret_20", "rel_ret_60"
            ]

            feats = df[feature_cols].dropna()

            # --- ADD TICKER AS SECOND INDEX LEVEL ---
            feats["Ticker"] = ticker
            feats = feats.set_index("Ticker", append=True)

            features_dict[ticker] = feats

        except Exception as e:
            print(f"[WARN] Feature build failed for {ticker}: {e}")

    # --- CONCAT ALL TICKERS ---
    if not features_dict:
        raise RuntimeError("No features built for any ticker")

    features = pd.concat(features_dict.values()).sort_index()

    # --- ENSURE INDEX NAMES ---
    features.index = features.index.set_names(["Date", "Ticker"])

    # --- CROSS-SECTIONAL RANKS ---
    for col in features.columns:
        features[f"{col}_rank"] = (
            features.groupby(level="Date")[col]
            .rank(pct=True)
        )

    return features

