# stock_data = yf.download(ticker, start="2000-01-01", end=current_date)
# data_source = "wrds"
# in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
# #Fetching stock price data
# data_dict = {ticker: yf.download(ticker, start="2024-01-01", end="2024-02-01") for ticker in tickers}
# Filter out tickers that failed to download
# valid_data = {ticker: df for ticker, df in data_dict.items() if not df.empty}
# """
# Param
# ----------
#     start_date : str
#         start date of the data
#     end_date : str
#         end date of the data
#     ticker_list : list
#         a list of stock tickers
# Example
# -------
# input:
# ticker_list = config_tickers.DOW_30_TICKER
# start_date = '2009-01-01'
# end_date = '2021-10-31'
# time_interval == "1D"

# output:
#     date	    tic	    open	    high	    low	        close	    volume
# 0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
# 1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
# 2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
# 3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
# ...
# """
# https://docs.alpaca.markets/docs/about-market-data-api#subscription-plans
import streamlit as st
from datetime import datetime
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET

def alpaca_history(alpaca_API_KEY, alpaca_API_SECRET):
  from alpaca.data.historical import StockHistoricalDataClient
  from alpaca.data.requests import StockBarsRequest
  from alpaca.data.timeframe import TimeFrame
  client = StockHistoricalDataClient(alpaca_API_KEY, alpaca_API_SECRET)
  request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL", "AXP"],  # Stock symbol(s) to fetch
    timeframe=TimeFrame.Day,            # Timeframe: Daily data
    start=datetime(2025, 1, 1),         # Start date
    end  =datetime(2025, 3, 16)         # End date
  )  
  btc_bars = client.get_stock_bars(request_params)
  return btc_bars

btc_bars = alpaca_history(ALPACA_API_KEY, ALPACA_API_SECRET)
if btc_bars:
    st.write(btc_bars.df)
else:
    st.write("⚠️ No data retrieved!")
