from data_loader import load_price_data, load_nifty
from ml.features import build_feature_matrix

price_data = load_price_data()
nifty = load_nifty()

X = build_feature_matrix(price_data, nifty)

# print(X.shape)
# print(X.columns.levels[1])
# print(X.dropna().shape)
print(type(X.index))
print(X.index.names)
print(X.columns[:10])
