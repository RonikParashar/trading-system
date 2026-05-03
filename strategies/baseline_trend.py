import pandas as pd

def generate_signals(df):
    entries = (df["Close"] > df["sma_20"]) & (df["sma_10"] > df["sma_20"])
    exits = df["Close"] < df["sma_20"]
    return entries, exits
