from data.live_data_loader import load_live_prices

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "^NSEI"]

price_data = load_live_prices(tickers)

print(price_data["RELIANCE.NS"].tail())
