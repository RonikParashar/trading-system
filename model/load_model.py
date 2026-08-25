"""
Load trained ML model for trading signal generation.
"""

import os
import pickle
from pathlib import Path


MODEL_DIR = Path("model")
DEFAULT_MODEL_FILE = MODEL_DIR / "model.pkl"


def load_model(model_path: str = None) -> object:
    """
    Load trained model for inference.

    Args:
        model_path: Path to model file (optional, defaults to model/model.pkl)

    Returns:
        Trained model instance
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_FILE

    model_path = Path(model_path)

    if model_path.exists():
        print(f"Loading model from: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model
    else:
        print(f"[WARN] Model file not found: {model_path}")
        print("Initializing default model for inference...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        return model
