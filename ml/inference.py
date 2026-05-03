import pandas as pd
import numpy as np
import joblib
from typing import Dict

from ml.features import build_feature_matrix
from ml.labels import build_labels


def run_daily_ml_inference(
    price_data: Dict[str, pd.DataFrame],
    model_path: str = "ml/model.pkl",
    benchmark_path: str = "ml/nifty.pkl",
) -> pd.DataFrame:
    """
    Daily ML Inference using walk-forward approach.

    Returns:
        scores DataFrame
        index: Date
        columns: tickers
    """

    # Load benchmark (NIFTY)
    try:
        benchmark = joblib.load(benchmark_path)
    except:
        # If benchmark not saved, download fresh
        from data.live_data_loader import load_live_prices
        from datetime import datetime, timedelta
        lookback = 400
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback)
        nifty_df = joblib.loads(
            joblib.dump(yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True), benchmark_path)
        )
        benchmark = nifty_df["Close"].pct_change()
        benchmark = benchmark.dropna()

    # Build features & labels
    try:
        X = build_feature_matrix(price_data, benchmark)
    except Exception as e:
        print(f"[ERROR] Feature build failed: {e}")
        raise

    # Build labels for prediction horizon
    y = build_labels(price_data, horizon=20)

    # Align indices
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]

    if X.empty:
        raise ValueError("No valid data for inference")

    # Walk-forward predictions
    from ml.walkforward import walkforward_train_predict
    preds = walkforward_train_predict(X, y)

    if preds.empty:
        print("[WARN] No predictions generated - insufficient training data")
        return pd.DataFrame()

    # Convert to (Date x Ticker) matrix
    scores = (
        preds
        .rename("score")
        .reset_index()
        .pivot(index="Date", columns="Ticker", values="score")
        .sort_index()
    )

    return scores
