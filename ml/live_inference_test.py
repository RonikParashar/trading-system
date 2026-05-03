from data.live_data_loader import load_live_prices
from ml.live_inference import run_daily_ml_inference

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
price_data = load_live_prices(tickers)

scores = run_daily_ml_inference(price_data)

print(scores)
