import pandas as pd
from ml.model import train_model, predict


def walkforward_train_predict(
    X,
    y,
    train_years=5,
    test_months=1,
    purge_days=20  # must match (or exceed) the forward-return horizon in ml/labels.py
):
    """
    Walk-forward ML training for cross-sectional data.

    purge_days: the last `purge_days` of each training window are dropped.
    Labels are forward-looking (return over the next `horizon` days), so
    without this, the tail of the training set contains labels computed
    from prices inside the test window -- i.e. leakage.
    """
    results = []

    dates = X.index.get_level_values("Date").unique()
    train_days = train_years * 252
    step = test_months * 21

    for i in range(train_days, len(dates), step):
        train_start = dates[i - train_days]
        test_start = dates[i - 1]              # last day before test begins (unchanged)
        train_end_idx = (i - 1) - purge_days   # purge: cut training short by purge_days
        test_end = dates[min(i + step - 1, len(dates) - 1)]

        if train_end_idx < i - train_days:
            continue  # not enough data left after purging

        train_end = dates[train_end_idx]

        train_mask = (
            (X.index.get_level_values("Date") >= train_start) &
            (X.index.get_level_values("Date") <= train_end)
        )

        test_mask = (
            (X.index.get_level_values("Date") > test_start) &
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