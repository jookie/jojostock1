import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models.bars import BarSet
from alpaca_trade_api import REST
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
from lib.utility.util import (
    get_real_time_price,
    fetch_stock_data,
    fetch_news_data,
    analyze_sentiment,
    display_sentiment_summary,
    plot_stock_data
)


FINVIZ_URL = "https://finviz.com/quote.ashx?t="

# 🔥 Download NLTK dependencies
nltk.download("vader_lexicon")

# 📡 Fetch Real-Time Stock Price
def get_real_time_price9999(ticker):
    api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url="https://data.alpaca.markets/v2")
    trade = api.get_latest_trade(ticker)
    return trade.price

# 🔄 Cached API Client
@st.cache_resource
def get_stock_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# 📊 Fetch Stock Data from Alpaca
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        client = get_stock_client()
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, start=start_date, end=end_date)
        return client.get_stock_bars(request_params)
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
        return None

# 📊 Convert Alpaca Data to DataFrame
def convert_alpaca_data_to_df(stock_data):
    """Converts Alpaca BarSet to a Pandas DataFrame for visualization."""
    if isinstance(stock_data, BarSet):
        data_list = []
        for symbol, bars in stock_data.data.items():
            for bar in bars:
                data_list.append({
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume
                })

        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame to avoid errors

# 📰 Fetch News Data from Finviz
@st.cache_data
def fetch_news_data(ticker):
    try:
        req = Request(url=FINVIZ_URL + ticker, headers={"user-agent": "Mozilla/5.0"})
        html = BeautifulSoup(urlopen(req), features="html.parser")
        news_table = html.find(id="news-table")

        if news_table:
            return [{"title": row.a.text, "date": row.td.text.split()[1], "time": row.td.text.split()[0]} 
                    for row in news_table.findAll("tr") if row.a]
    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return None

# 🧠 Analyze Sentiment from News
@st.cache_data
def analyze_sentiment(news_list):
    if not news_list:
        return pd.DataFrame()  # Return empty DataFrame if no news

    df_sentiment = pd.DataFrame(news_list)
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["date"], errors="coerce", utc=True)
    df_sentiment["Compound Score"] = df_sentiment["title"].apply(lambda title: SentimentIntensityAnalyzer().polarity_scores(title)["compound"])
    df_sentiment.set_index("Date", inplace=True)
    return df_sentiment

# 📊 Display Sentiment Summary
def display_sentiment_summary(df_sentiment):
    st.subheader("📊 Sentiment Summary")
    avg_score = df_sentiment["Compound Score"].mean() * 100
    sentiment_icon = "😊" if avg_score > 10 else "😢" if avg_score < -10 else "😐"
    st.metric(label=f"💡 Average Sentiment Score {sentiment_icon}", value=f"{avg_score:.2f}%")

    summary = {
        "📈 Positive": f"{(df_sentiment['Compound Score'] > 0).sum() / len(df_sentiment) * 100:.2f}%",
        "📉 Negative": f"{(df_sentiment['Compound Score'] < 0).sum() / len(df_sentiment) * 100:.2f}%",
        "⚖️ Neutral": f"{(df_sentiment['Compound Score'] == 0).sum() / len(df_sentiment) * 100:.2f}%"
    }
    st.json(summary)

# 📈 Plot Stock Data
def plot_stock_data(stock_data, ticker):
    """Processes stock data and plots it."""
    df_stock = convert_alpaca_data_to_df(stock_data)

    if not df_stock.empty:
        st.subheader(f"📈 {ticker} Stock Price Movements")
        st.line_chart(df_stock[["close"]])
    else:
        st.warning("⚠️ No valid stock data available.")
