import pandas as pd
from ml.model import train_model, predict


def walkforward_train_predict(
    X,
    y,
    train_years=5,
    test_months=1
):
    """
    Walk-forward ML training for cross-sectional data
    """
    results = []

    dates = X.index.get_level_values("Date").unique()
    train_days = train_years * 252
    step = test_months * 21

    for i in range(train_days, len(dates), step):
        train_start = dates[i - train_days]
        train_end = dates[i - 1]
        test_end = dates[min(i + step - 1, len(dates) - 1)]

        # --- IMPORTANT FIX ---
        train_mask = (
            (X.index.get_level_values("Date") >= train_start) &
            (X.index.get_level_values("Date") <= train_end)
        )

        test_mask = (
            (X.index.get_level_values("Date") > train_end) &
            (X.index.get_level_values("Date") <= test_end)
        )

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]

        X_test = X.loc[test_mask]

        if len(X_train) < 1000:
            continue

        model = train_model(X_train, y_train)
        preds = predict(model, X_test)

        results.append(preds)

    return pd.concat(results).sort_index()
