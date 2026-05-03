from data_loader import load_price_data, load_nifty
from ml.features import build_feature_matrix
from ml.labels import build_labels
from ml.walkforward import walkforward_train_predict
from ml.metrics import information_coefficient, rank_ic


def main():
    price_data = load_price_data()
    nifty = load_nifty()

    X = build_feature_matrix(price_data, nifty)
    y = build_labels(price_data)

    # ---- CRITICAL ALIGNMENT STEP ----
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]


    preds = walkforward_train_predict(X, y)

    ic = information_coefficient(preds, y)
    ric = rank_ic(preds, y)

    print("\n--- ML ALPHA METRICS ---")
    print(f"Mean IC      : {ic.mean():.4f}")
    print(f"Mean Rank IC : {ric.mean():.4f}")
    print(f"IC > 0 %     : {(ic > 0).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
