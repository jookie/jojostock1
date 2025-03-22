# 📈 Stock Sentiment Analysis & Trading Strategy
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models.bars import BarSet
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
# from lib.rl.config import DOW_30_TICKER
from lib.rl.config_tickers import DOW_30_TICKER
from lib.rl.config_tickers import index_dict

from lib.utility.util import (
    get_ticker_start_end_date,
    get_real_time_price,
    fetch_stock_data,
    fetch_news_data,
    analyze_sentiment,
    display_sentiment_summary,
    plot_stock_data,
    compute_moving_averages,
    generate_trade_signals,
    convert_barSet_to_DataFrame,
    compute_moving_averages,
    collapsible_detailed_description,
    load_and_plot_stock_data,
)

# 🔥 Download NLTK dependencies
nltk.download("vader_lexicon")

# 🎨 Streamlit UI Configuration
st.set_page_config(page_title="Stock Sentiment & Trading", layout="wide", page_icon="📊")
st.markdown(
    """
    <style>
        .block-container { padding-top: 20px; }
        h1 { color: #3498db; text-align: center; }
        h2 { color: #e74c3c; }
        .stAlert { font-size: 18px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 📌 Define available tickers
TICKERS  = DOW_30_TICKER
FINVIZ_URL = "https://finviz.com/quote.ashx?t="

# 🛠️ Layout for ticker & date selection in a single row
st.markdown("## 📊 Stock Sentiment & Trading Strategy")

header = "📖 Analyze stock trends using real-time market data, historical price movements, and AI-powered sentiment analysis."
content =  """This application combines **news sentiment analysis** and **historical stock data** 
        to provide insights into stock market trends. It helps traders make data-driven 
        decisions by analyzing **price movements** and **market sentiment**.

        ### **Key Features**
        - 📡 **Live Stock Price Updates** from Alpaca API.
        - 📰 **News Sentiment Analysis** using NLP (VADER).
        - 📊 **Buy/Sell Signals & Moving Averages** (SMA/EMA).
        - 🤖 **AI-Powered Sentiment Analysis** (GPT-4 integration).
        - 💰 Portfolio Analysis – Track multiple stocks and calculate profit/loss.
        - 🤖 **Auto-Trading Support** via Alpaca API.
        - 🔮 Stock Prediction – Use AI models like LSTMs to predict future prices.
        - 🌍 Alternative Data Sources – Integrate Twitter, Reddit, or Google Trends.
        - 📰 Daily Market Summary – Show top gainers/losers and market trends.


        🚀 **Select a stock ticker and a date range to begin!**
        """
# Collapsible Detailed Description
collapsible_detailed_description(header, content)

ticker, start_date, end_date = get_ticker_start_end_date(index_dict)

# 🏆 Main App Logic
if ticker:
   df_stock_ , close_col = load_and_plot_stock_data(ticker, start_date, end_date)

if not df_stock_:  # ✅ Correct way to check if BarSet is empty
    st.warning(f"⚠️ No stock data found for {ticker}. Try another ticker or date range.")
else :
    if df_stock_:
        st.subheader(f"📊 {ticker} Stock Data from {start_date} to {end_date}")
        plot_stock_data(df_stock_, ticker)
    else:
        st.warning("⚠️ No stock data retrieved. Try adjusting the date range or ticker.")
    # 📰 News Section
    news_data = fetch_news_data(ticker)
    if news_data:
        df_news = analyze_sentiment(news_data)
        if df_news is not None:
            st.subheader("📰 Latest News & Sentiment Scores")
            st.dataframe(df_news)
            display_sentiment_summary(df_news)
        else:
            st.warning("⚠️ No sentiment data available.")

    # Ensure correct 'close' column
    compute_moving_averages(df_stock_, ticker, df_news)
        # Generate buy/sell signals
    df_merged = generate_trade_signals(df_stock_, df_news, close_col)

    # Plot results
    st.line_chart(df_merged[[close_col, "SMA_50", "EMA_20", "Buy_Signal", "Sell_Signal"]])
