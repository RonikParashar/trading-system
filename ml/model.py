import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_model():
    """
    Ridge regression is robust, stable, and alpha-friendly
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])


def train_model(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return pd.Series(
        model.predict(X),
        index=X.index,
        name="prediction"
    )
