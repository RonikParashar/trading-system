from data_loader import load_price_data
from ml.labels import build_labels_for_universe

price_data = load_price_data()
labels = build_labels_for_universe(price_data)

print(labels.describe())
print(labels.isna().mean().sort_values().head())
