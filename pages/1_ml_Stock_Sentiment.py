# Build a sentiment analysis-based trading strategy using Streamlit for the user interface and LumiBot for trading.
# This script integrates Streamlit for visualization, VADER for sentiment analysis, and Alpaca for stock data retrieval.

from lib.rl.meta.data_processor import DataProcessor
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import datetime
from alpaca.data.models.bars import BarSet

# Download NLTK dependencies
nltk.download('vader_lexicon')

# Custom CSS for Streamlit UI
custom_css = """
<style>
body {
background-color: black;
font-family: "Times New Roman", Times, serif;
color: white;
line-height: 1.6;
}
h1 {
color: #3498db;
}
h2 {
color: #e74c3c;
}
p {
margin: 10px 0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Define available tickers
finviz_url = "https://finviz.com/quote.ashx?t="
example_ticker_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "JPM", "NFLX", "FB", "BRK.B", "V",
    "NVDA", "DIS", "BA", "IBM", "GE",
    "PG", "JNJ", "KO", "MCD", "T",
    "ADBE", "CRM", "INTC", "ORCL", "HD"
]

ticker = st.selectbox("Select a stock ticker symbol or enter your own:", example_ticker_symbols)

def alpaca_history(alpaca_API_KEY, alpaca_API_SECRET):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    client = StockHistoricalDataClient(alpaca_API_KEY, alpaca_API_SECRET)
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Day,
        start=datetime.datetime(2025, 1, 1),
        end=datetime.datetime(2025, 3, 16)
    )  
    return client.get_stock_bars(request_params)

news_tables = {}
if ticker:
    from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
    stock_data = alpaca_history(ALPACA_API_KEY, ALPACA_API_SECRET)
    
    if isinstance(stock_data, BarSet):
        df_list = []
        for symbol in stock_data.data.keys():
            bars = stock_data[symbol]
            if isinstance(bars, list) and len(bars) > 0:
                df = pd.DataFrame([bar.dict() for bar in bars])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df.add_prefix(f"{symbol}_")
                if not df.empty:
                    df_list.append(df)
        
        if df_list:
            combined_data = pd.concat(df_list, axis=1)
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
            st.write(combined_data.head())
            
            # Fixing plotting by ensuring correct column selection
            plt.figure(figsize=(10, 6))
            close_col = f"{ticker}_close" if f"{ticker}_close" in combined_data.columns else None
            
            if close_col:
                plt.plot(combined_data.index, combined_data[close_col], label=ticker)
                plt.xlabel("Date")
                plt.ylabel("Stock Price")
                plt.title(f"{ticker} Stock Price Movements")
                plt.legend()
                plt.xticks(rotation=45)
                st.pyplot(plt)
            else:
                st.write(f"⚠️ No closing price data available for {ticker}.")
        else:
            st.write("⚠️ No valid stock data retrieved! Check API or ticker availability.")
    else:
        st.write("⚠️ Unexpected data structure received from Alpaca API.")
    
    # Fetch news data
    url = finviz_url + ticker
    req = Request(url=url, headers={"user-agent": "my-app"})
    response = urlopen(req)
    html = BeautifulSoup(response, features="html.parser")
    news_table = html.find(id="news-table")
    news_tables[ticker] = news_table

if news_table:
      parsed_data = []
      for ticker, news_table in news_tables.items():
            for row in news_table.findAll('tr'):
                  if row.a:
                        title = row.a.text
                        date_data = row.td.text.split()
                        date = date_data[1] if len(date_data) > 1 else "Unknown"
                        time = date_data[0]
                        parsed_data.append([ticker, date, time, title])

      df = pd.DataFrame(parsed_data, columns=["Ticker", "Date", "Time", "Headline"])
      vader = SentimentIntensityAnalyzer()
      df["Compound Score"] = df["Headline"].apply(lambda title: vader.polarity_scores(title)["compound"])
      df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
      
      st.subheader("News Headlines and Sentiment Scores")
      st.dataframe(df)
      
      sentiment_summary = {
            "Average Score": df["Compound Score"].mean(),
            "Positive": (df["Compound Score"] > 0).sum() / len(df) * 100,
            "Negative": (df["Compound Score"] < 0).sum() / len(df) * 100,
            "Neutral": (df["Compound Score"] == 0).sum() / len(df) * 100,
      }
      st.subheader("Sentiment Summary")
      st.write(sentiment_summary)

      plt.figure(figsize=(10, 8))
      # dov to numpyh
      # The error indicates that Pandas is no longer supporting multi-dimensional indexing directly on its objects, such as trying to index a DataFrame or Series with [:, None]. Instead, you need to convert the object to a NumPy array before performing such indexing.
      # plt.plot(stock_data.index.to_numpy(), stock_data["Close"])
      # # plt.plot(stock_data.index[:, None], stock_data["Close"])
      # plt.xlabel("Date")
      # plt.ylabel("Stock Price")
      # plt.title("Stock Price Movements - Line Chart")
      # plt.xticks(rotation=45)
      # st.subheader("Jojo Price Movements - JOJO Chart")
      # st.pyplot(plt)
else:
    st.write("No news found for the entered stock ticker symbol.")