import pandas as pd
import yfinance as yf
from datetime import datetime

# Define tickers and date range
tickers = ["AAPL", "AXP"]  # Add your tickers
start_date = "2025-01-01"
end_date = "2025-07-01"

# Download the stock data
data_dict = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

# Filter out tickers that failed to download
valid_data = {ticker: df for ticker, df in data_dict.items() if not df.empty}

if valid_data:
    combined_data = pd.concat(valid_data.values(), axis=1, keys=valid_data.keys())
    print(combined_data.head())
else:
    print("⚠️ No valid data retrieved! Check ticker names or availability.")
