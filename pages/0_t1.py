# 📈 Stock Sentiment Analysis & Trading Strategy
import datetime
import pandas as pd
import streamlit as st
from lib.utility.util import (
    get_real_time_price,
    fetch_stock_data,
    fetch_news_data,
    analyze_sentiment,
    display_sentiment_summary,
    plot_stock_data
)
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET

# 🎨 Streamlit UI Configuration
st.set_page_config(page_title="Stock Sentiment & Trading", layout="wide", page_icon="📊")

# 📌 Define available tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "NFLX", "FB", "BRK.B", "V",
           "NVDA", "DIS", "BA", "IBM", "GE", "PG", "JNJ", "KO", "MCD", "T",
           "ADBE", "CRM", "INTC", "ORCL", "HD"]

# 📌 Title & One-Line Description
st.markdown("## 📊 Stock Sentiment & Trading Strategy")
st.markdown("Analyze stock trends using real-time market data, historical price movements, and AI-powered sentiment analysis.")

# 📖 Expandable Detailed Description
with st.expander("📖 About..."):
    st.markdown("""
    This application provides **a data-driven approach to stock market analysis** 
    by combining **real-time stock data, historical price trends, and AI-powered sentiment analysis** 
    from financial news. It helps traders and investors make informed decisions based on **technical indicators** 
    and **market sentiment**.

    ### **🔹 Key Features**
    - 📡 **Live Stock Price Updates** from Alpaca API.
    - 📰 **News Sentiment Analysis** using NLP (VADER).
    - 📊 **Buy/Sell Signals & Moving Averages** (SMA/EMA).
    - 🤖 **AI-Powered Sentiment Analysis** (GPT-4 integration).
    - 📈 **Portfolio Tracking** with profit/loss calculation.
    - 🤖 **Auto-Trading Support** via Alpaca API.

    🚀 **Select a stock ticker and a date range to begin!**
    """)

# 🏆 UI Layout: Ticker & Date Selection
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.selectbox("📈 Select a Stock Ticker:", TICKERS)
with col2:
    start_date = st.date_input("📅 Start Date", datetime.date(2024, 1, 1))
with col3:
    end_date = st.date_input("📅 End Date", datetime.date.today())

# 📡 Live Stock Price
st.subheader("📡 Live Stock Price")
latest_price = get_real_time_price(ticker)
st.metric(label=f"{ticker} Price", value=f"${latest_price:.2f}")

# 🏆 Main App Logic
stock_data = fetch_stock_data(ticker, start_date, end_date)

if stock_data is not None:
    df_stock = plot_stock_data(stock_data, ticker)

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
else:
    st.warning("⚠️ No stock data retrieved. Try adjusting the date range or ticker.")
