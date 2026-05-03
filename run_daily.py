from data.live_data_loader import load_live_prices
from ml.inference import run_daily_inference
from paper_trading.paper_trade_runner import run_daily_paper_trade

TODAY = "2026-01-15"

price_data = load_live_prices()
scores = run_daily_inference(price_data)
benchmark = price_data["NIFTY"]

run_daily_paper_trade(
    date=TODAY,
    price_data=price_data,
    scores=scores,
    benchmark=benchmark,
)
