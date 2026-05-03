# backtest/check_backtest.py
import pandas as pd
import vectorbt as vbt
from pathlib import Path
from strategies.baseline_trend import generate_signals

DATA_DIR = Path("data/processed")
TICKERS = ["RELIANCE.NS", "AXISBANK.NS"]   # change/add tickers as needed

def safe_stats_to_print(pf):
    """Return a pandas Series of key stats (string-friendly)."""
    s = pf.stats()
    # keep only a few important rows to avoid huge output
    keys = [
        "Start", "End", "Period", "Start Value", "End Value", "Total Return [%]",
        "Benchmark Return [%]", "Max Drawdown [%]", "Total Trades", "Total Closed Trades",
        "Win Rate [%]", "Profit Factor", "Sharpe Ratio", "Total Fees Paid"
    ]
    out = pd.Series({k: s.get(k, "N/A") for k in keys})
    return out

def run_checks_for_ticker(ticker):
    file = DATA_DIR / f"{ticker}.csv"
    if not file.exists():
        print(f"[ERROR] {file} not found. Skipping {ticker}.")
        return

    print("\n" + "="*40)
    print(f"CHECKS FOR {ticker}")
    print("="*40)

    df = pd.read_csv(file, parse_dates=["Date"], index_col="Date")
    # basic sanity
    print("Data range:", df.index.min(), "to", df.index.max())
    print("Rows:", len(df))
    if (df["Close"] <= 0).any():
        print("WARNING: Non-positive close prices found!")

    # 1) regenerate signals
    entries, exits = generate_signals(df)
    print(f"Signals generated -> Entries: {entries.sum()}, Exits: {exits.sum()}")

    # 2) portfolio using close prices (how you originally ran it)
    pf_close = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        fees=0.001,
        slippage=0.001,
        init_cash=100000,
        freq="1D"
    )
    stats_close = safe_stats_to_print(pf_close)
    print("\n--- STATS (execute on CLOSE with current signals) ---")
    print(stats_close.to_string())

    # show first 5 trades (readable)
    tr = pf_close.trades.records_readable
    if tr is None or tr.empty:
        print("\nNo trades were generated (close-based).")
    else:
        print("\nFirst 5 trades (close-based):")
        print(tr.head().to_string())

    # 3) manual buy-and-hold (benchmark check)
    try:
        bh_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        print(f"\nManual Buy & Hold return [%]: {bh_return:.2f}")
    except Exception as e:
        print("Error computing buy-and-hold:", e)

    # 4) Execute using NEXT-DAY OPEN (conservative, no look-ahead)
    entries_shift = entries.shift(1).fillna(False)
    exits_shift = exits.shift(1).fillna(False)

    # make sure shifted entries are not accidentally all False
    print("Shifted entries sum (entries.shift(1)):", entries_shift.sum())

    pf_next_open = vbt.Portfolio.from_signals(
        close=df["Open"],              # execute at next-day open
        entries=entries_shift,
        exits=exits_shift,
        fees=0.001,
        slippage=0.001,
        init_cash=100000,
        freq="1D"
    )
    stats_open = safe_stats_to_print(pf_next_open)
    print("\n--- STATS (execute on NEXT-DAY OPEN with shifted signals) ---")
    print(stats_open.to_string())

    # 5) compare trade lists and fees
    tr_open = pf_next_open.trades.records_readable
    print("\nTrades (next-open) - first 5 rows:")
    if tr_open is None or tr_open.empty:
        print("No trades (next-open).")
    else:
        print(tr_open.head().to_string())

    # fees checks
    sc = pf_close.stats()
    so = pf_next_open.stats()
    fees_close = sc.get("Total Fees Paid", None)
    fees_open = so.get("Total Fees Paid", None)
    print(f"\nTotal Fees Paid (close-exec): {fees_close}")
    print(f"Total Fees Paid (next-open): {fees_open}")

    # avg fee per closed trade (best-effort)
    closed_close = 0
    closed_open = 0
    if tr is not None and not tr.empty:
        # entry rows with non-null Exit Date count as closed trades
        closed_close = tr[tr["Exit Date"].notna()].shape[0]
    if tr_open is not None and not tr_open.empty:
        closed_open = tr_open[tr_open["Exit Date"].notna()].shape[0]

    if fees_close not in (None, "N/A") and closed_close > 0:
        print(f"Avg fee per closed trade (close-exec): {fees_close / closed_close:.2f}")
    if fees_open not in (None, "N/A") and closed_open > 0:
        print(f"Avg fee per closed trade (next-open): {fees_open / closed_open:.2f}")

    # 6) quick look-ahead heuristic:
    # If performance on CLOSE is *much* better than NEXT-DAY OPEN, that suggests look-ahead / same-day execution advantage.
    try:
        tr_close_pct = stats_close["Total Return [%]"]
        tr_open_pct = stats_open["Total Return [%]"]
        diff = float(tr_close_pct) - float(tr_open_pct)
        print(f"\nDelta Total Return (close_exec - nextopen_exec) [%]: {diff:.2f}")
        if diff > 5.0:
            print("NOTICE: Close-based execution produces noticeably higher returns than next-open. This may indicate look-ahead (same-day) execution in the close-based run.")
    except Exception:
        pass

    # print a short summary line
    print("\nSummary (short):")
    print(f" - Ticker: {ticker}")
    print(f" - Trades (close-based): {len(tr) if tr is not None else 0}")
    print(f" - Trades (next-open): {len(tr_open) if tr_open is not None else 0}")
    print(f" - Buy&Hold [%]: {bh_return:.2f}")
    print("-"*40)

def main():
    for t in TICKERS:
        run_checks_for_ticker(t)

if __name__ == "__main__":
    main()
