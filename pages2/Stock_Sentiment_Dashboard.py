# 📈 Unified Stock Sentiment Dashboard
import datetime
import pandas as pd
import streamlit as st
from lib.rl.config_tickers import DOW_30_TICKER, index_dict
from lib.utility.util import (
    get_real_time_price,
    fetch_stock_data,
    fetch_news_data,
    analyze_sentiment,
    display_sentiment_summary,
    plot_stock_data,
    compute_moving_averages,
    generate_trade_signals,
    convert_barSet_to_DataFrame,
    fetch_twitter_sentiment,
    fetch_reddit_sentiment,
    fetch_google_trends,
    collapsible_detailed_description
)

# UI Setup
st.set_page_config(page_title="📊 Stock Sentiment Dashboard", layout="wide")
st.title("📈 Stock Sentiment Analysis & Strategy Dashboard")

collapsible_detailed_description(
    "📖 About...",
    "📊 This dashboard combines real-time stock data, news sentiment, technical indicators, and alternative signals "
    "from social media and search trends to help you make informed decisions."
)

# --- Ticker & Date Inputs ---
idx_col1, idx_col2, date_col1, date_col2 = st.columns(4)
with idx_col1:
    selected_index = st.selectbox("📂 Index:", list(index_dict.keys()))
    tickers = index_dict[selected_index]
with idx_col2:
    ticker = st.selectbox("💹 Stock Ticker:", tickers)
with date_col1:
    start_date = st.date_input("📅 Start Date", datetime.date(2024, 1, 1))
with date_col2:
    end_date = st.date_input("📅 End Date", datetime.date.today())

st.divider()

# --- Live Price Display ---
st.subheader("📡 Real-Time Price")
latest_price = get_real_time_price(ticker)
if latest_price:
    st.metric(label=f"{ticker} Price", value=f"${latest_price:.2f}")
else:
    st.warning("No real-time price available.")

# --- Fetch Stock Data ---
stock_data = fetch_stock_data(ticker, start_date, end_date)
if not stock_data or not stock_data.__dict__:
    st.warning("⚠️ No stock data found.")
    st.stop()
    
df_stock, close_col = convert_barSet_to_DataFrame(stock_data, None, ticker)

if df_stock is None or close_col is None or close_col not in df_stock.columns:
    st.error(f"⚠️ No closing price data available for {close_col}.")
    st.info(f"Try a more liquid ticker — {ticker} returned no usable closing data.")
    st.stop()

st.write("🔍 Raw stock_data:", stock_data)
st.write("🔍 Converted df_stock:", df_stock)
st.write("🔍 Columns:", df_stock.columns if df_stock is not None else "None")

# Continue...

# --- Plot Stock Price ---
plot_stock_data(stock_data, ticker)

# --- News Sentiment ---
st.subheader("📰 News Sentiment Analysis")
news_data = fetch_news_data(ticker)
df_news = analyze_sentiment(news_data) if news_data else pd.DataFrame()

if not df_news.empty:
    st.dataframe(df_news)
    display_sentiment_summary(df_news)
else:
    st.warning("⚠️ No sentiment data available.")

# --- Technicals ---
st.subheader("📈 Technical Analysis")
df_stock = compute_moving_averages(df_stock, close_col, df_news)
df_signals = generate_trade_signals(df_stock, df_news, close_col)

if not df_signals.empty:
    st.line_chart(df_signals[[close_col, "SMA_50", "EMA_20", "Buy_Signal", "Sell_Signal"]])

# --- Alternative Sentiment Sources ---
st.subheader("🌍 Alternative Data Sources")
alt_col1, alt_col2, alt_col3 = st.columns(3)
with alt_col1:
    if st.checkbox("🐦 Twitter"):
        st.write(fetch_twitter_sentiment(ticker))
with alt_col2:
    if st.checkbox("👽 Reddit"):
        st.write(fetch_reddit_sentiment(ticker))
with alt_col3:
    if st.checkbox("📈 Google Trends"):
        df_trend = fetch_google_trends(ticker)
        if not df_trend.empty:
            st.line_chart(df_trend.set_index("date"))
        else:
            st.info("No trends data available.")

st.success("✅ Dashboard loaded successfully.")
