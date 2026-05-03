import pandas as pd
from ml.features import build_feature_matrix
from ml.labels import build_labels
from ml.walkforward import walkforward_train_predict

def generate_ml_scores(price_data, nifty):
    """
    Returns:
        DataFrame: index=Date, columns=Tickers, values=ML scores
    """

    # Build features & labels
    X = build_feature_matrix(price_data, nifty)
    y = build_labels(price_data)

    # Align
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]

    # Walk-forward ML predictions
    preds = walkforward_train_predict(X, y)

    # Convert to (Date x Ticker) matrix
    scores = (
        preds
        .rename("score")
        .reset_index()
        .pivot(index="Date", columns="Ticker", values="score")
        .sort_index()
    )

    return scores
