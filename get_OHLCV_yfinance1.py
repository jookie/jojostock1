
from alpaca.data.historical import CryptoHistoricalDataClient


import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
current_date = datetime.now().strftime("%Y-%m-%d")
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL

import yfinance as yf
import streamlit as st
from datetime import datetime

# Define the ticker and date range
tickers = ["AAPL"]
start_date = "2022-09-01"
end_date = "2022-09-07"

# Fetch data from Yahoo Finance
data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

# Filter out tickers that failed to download
valid_data = {ticker: df for ticker, df in data.items() if not df.empty}

if valid_data:
    # Combine the data into a single DataFrame
    combined_data = pd.concat(valid_data.values(), axis=1, keys=valid_data.keys())
    st.write(combined_data)  # Display the DataFrame in Streamlit
else:
    st.write("⚠️ No valid data retrieved! Check ticker names or API availability.")
