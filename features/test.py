import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/HDFCBANK.NS.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

plt.figure(figsize=(12,5))
plt.plot(df["Close"], label="Close")
plt.plot(df["sma_20"], label="SMA20")
plt.legend()
plt.title("HDFCBANK Close vs SMA20")
plt.show()
