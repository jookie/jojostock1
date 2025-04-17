import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET

# Initialize Alpaca Data Client
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]
start_date = date.today() - timedelta(days=10)
end_date   = date.today() - timedelta(days=1)

print(f"Fetching data from {start_date} to {end_date}")

# Prepare dictionary to store data
data_frames = {}

for ticker in tickers:
    print(f"Fetching: {ticker}")
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = client.get_stock_bars(request_params).df

        if bars.empty:
            print(f"⚠️ No data for {ticker}")
            continue

        # Normalize index to ensure it's flat datetime, not tuple
        if isinstance(bars.index, pd.MultiIndex):
            bars.reset_index(inplace=True)
            bars.set_index("timestamp", inplace=True)

        df = bars[["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Convert index to datetime.date safely
        df.index = pd.to_datetime(df.index).date

        data_frames[ticker] = df

    except Exception as e:
        print(f"❌ Failed to fetch {ticker}: {e}")

# Combine into MultiIndex DataFrame
if data_frames:
    combined_df = pd.concat(data_frames.values(), axis=1, keys=data_frames.keys())
    print(combined_df.head())

    # Optional: Save to CSV
    combined_df.to_csv("alpaca_6_months_data.csv")
    print("📁 Saved to alpaca_6_months_data.csv")
else:
    print("🚫 No data returned for any ticker.")
