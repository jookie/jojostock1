from lumibot.backtesting import YahooDataBacktesting
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
current_date = datetime.now().strftime("%Y-%m-%d")

tickers = ["AXP"]  # Adjust if needed
data_dict = {ticker: yf.download(ticker, start="2024-01-01", end="2024-02-01") for ticker in tickers}

# Filter out tickers that failed to download
valid_data = {ticker: df for ticker, df in data_dict.items() if not df.empty}

if valid_data:
    combined_data = pd.concat(valid_data.values(), axis=1)
    print(combined_data.head())
else:
    print("⚠️ No valid data retrieved! Check ticker names or API availability.")
    
    

