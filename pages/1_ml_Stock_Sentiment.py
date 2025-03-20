# 📈 Stock Sentiment Analysis & Trading Strategy
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models.bars import BarSet
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET

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
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "NFLX", "FB", "BRK.B", "V",
    "NVDA", "DIS", "BA", "IBM", "GE", "PG", "JNJ", "KO", "MCD", "T",
    "ADBE", "CRM", "INTC", "ORCL", "HD"
]
FINVIZ_URL = "https://finviz.com/quote.ashx?t="

# 🛠️ Layout for ticker & date selection in a single row
st.markdown("## 📊 Stock Sentiment & Trading Strategy")

# Collapsible Detailed Description
with st.expander("📖 Analyze stock trends using real-time market data, historical price movements, and AI-powered sentiment analysis."):
    st.markdown("""
    This application combines **news sentiment analysis** and **historical stock data** 
    to provide insights into stock market trends. It helps traders make data-driven 
    decisions by analyzing **price movements** and **market sentiment**.

    ### **Key Features**
    - 📡 **Live Stock Price Updates** from Alpaca API.
    - 📰 **News Sentiment Analysis** using NLP (VADER).
    - 📊 **Buy/Sell Signals & Moving Averages** (SMA/EMA).
    - 🤖 **AI-Powered Sentiment Analysis** (GPT-4 integration).
    - 📈 **Portfolio Tracking** with profit/loss calculation.
    - 🤖 **Auto-Trading Support** via Alpaca API.

    🚀 **Select a stock ticker and a date range to begin!**
    """)



col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker = st.selectbox("📈 Select a Stock Ticker:", TICKERS)

with col2:
    start_date = st.date_input("📅 Start Date", datetime.date(2024, 1, 1))

with col3:
    end_date = st.date_input("📅 End Date", datetime.date.today())
    
import time
col1, col2 = st.columns([2, 1])
 
from alpaca_trade_api import REST

def get_real_time_price(ticker):
    """Fetches real-time stock price from Alpaca API."""
    api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url="https://data.alpaca.markets/v2")
    trade = api.get_latest_trade(ticker)
    return trade.price  # ✅ Correct attribute for latest price

st.subheader("📡 Live Stock Price")
latest_price = get_real_time_price(ticker)
st.metric(label=f"{ticker} Price", value=f"${latest_price:.2f}")

# 🔄 Cached API Client
@st.cache_resource
def get_stock_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# 📊 Fetch Stock Data from Alpaca
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        client = get_stock_client()
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, start=start_date, end=end_date
        )
        return client.get_stock_bars(request_params)
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
        return None

# 📰 Fetch News Data from Finviz
@st.cache_data
def fetch_news_data(ticker):
    try:
        req = Request(url=FINVIZ_URL + ticker, headers={"user-agent": "Mozilla/5.0"})
        html = BeautifulSoup(urlopen(req), features="html.parser")
        news_table = html.find(id="news-table")

        if news_table:
            news_list = []
            for row in news_table.findAll("tr"):
                if row.a:
                    title = row.a.text
                    date_data = row.td.text.split()
                    date, time = (date_data[1], date_data[0]) if len(date_data) > 1 else ("Unknown", "")
                    news_list.append({"title": title, "date": date, "time": time})

            return news_list  # ✅ Return as a serializable list of dictionaries
    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return None

# 🧠 Analyze Sentiment from News
@st.cache_data
def analyze_sentiment(news_list, ticker):
    if not news_list:
        return None

    df_sentiment = pd.DataFrame(news_list)
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["date"], errors="coerce").dt.date
    df_sentiment["Compound Score"] = df_sentiment["title"].apply(lambda title: SentimentIntensityAnalyzer().polarity_scores(title)["compound"])
    return df_sentiment

# 📊 Display Sentiment Summary with Icons
def display_sentiment_summary(df_sentiment):
    st.subheader("📊 Sentiment Summary")

    # Calculate sentiment values
    avg_score = df_sentiment["Compound Score"].mean() * 100
    positive_pct = (df_sentiment["Compound Score"] > 0).sum() / len(df_sentiment) * 100
    negative_pct = (df_sentiment["Compound Score"] < 0).sum() / len(df_sentiment) * 100
    neutral_pct = (df_sentiment["Compound Score"] == 0).sum() / len(df_sentiment) * 100

    # Determine the appropriate icon based on sentiment
    if avg_score > 10:
        sentiment_icon = "😊"  # Positive sentiment
    elif avg_score < -10:
        sentiment_icon = "😢"  # Negative sentiment
    else:
        sentiment_icon = "😐"  # Neutral sentiment

    # Display Average Score with an Icon
    st.metric(label=f"💡 Average Sentiment Score {sentiment_icon}", value=f"{avg_score:.2f}%")

    # Display Sentiment Breakdown
    summary = {
        "📈 Positive": f"{positive_pct:.2f}%",
        "📉 Negative": f"{negative_pct:.2f}%",
        "⚖️ Neutral": f"{neutral_pct:.2f}%"
    }
    st.json(summary)

# 📈 Plot Stock Data
def plot_stock_data(data, ticker):
    if isinstance(data, BarSet):
        df_list = []
        for symbol, bars in data.data.items():
            if isinstance(bars, list) and bars:
                df_stock = pd.DataFrame([bar.dict() for bar in bars])
                df_stock["timestamp"] = pd.to_datetime(df_stock["timestamp"])
                df_stock.set_index("timestamp", inplace=True)
                df_stock = df_stock.add_prefix(f"{symbol}_")
                df_list.append(df_stock)
        
        if df_list:
            combined_data = pd.concat(df_list, axis=1).loc[:, ~pd.concat(df_list, axis=1).columns.duplicated()]
            close_col = f"{ticker}_close" if f"{ticker}_close" in combined_data.columns else None
            
            if close_col:
                st.subheader(f"📈 {ticker} Stock Price Movements")
                st.line_chart(combined_data[close_col])
            else:
                st.warning(f"⚠️ No closing price data available for {ticker}.")
        else:
            st.warning("⚠️ No valid stock data retrieved! Check API or ticker availability.")
    else:
        st.warning("⚠️ Unexpected data structure received from Alpaca API.")

# 🏆 Main App Logic
if ticker:
    stock_data = fetch_stock_data(ticker, start_date, end_date)

# Convert BarSet object to DataFrame
if isinstance(stock_data, BarSet):
    
    df_list = []
    for symbol, bars in stock_data.data.items():
        if isinstance(bars, list) and bars:
            df_stock = pd.DataFrame([bar.dict() for bar in bars])
            df_stock["timestamp"] = pd.to_datetime(df_stock["timestamp"])
            df_stock.set_index("timestamp", inplace=True)
            df_stock = df_stock.add_prefix(f"{symbol}_")
            df_list.append(df_stock)

            # DOV debug   
            # st.write("📊 DataFrame Columns:", df_stock.columns)
            if df_list:
                                                   
                df_stock_ = pd.concat(df_list, axis=1).loc[:, ~pd.concat(df_list, axis=1).columns.duplicated()]
                
                # 🔍 Debugging: Check column names
                st.write("📊 DataFrame Structure:", df_stock_.head())

                # Ensure we reference the correct column name
                close_col = f"{ticker}_close" if f"{ticker}_close" in df_stock_.columns else None

                if close_col:
                    # ✅ Compute Moving Averages
                    df_stock_["SMA_50"] = df_stock_[close_col].rolling(window=50).mean()
                    df_stock_["EMA_20"] = df_stock_[close_col].ewm(span=20, adjust=False).mean()

                    # ✅ Plot Data
                    st.subheader("📊 Moving Averages & Buy/Sell Signals")
                    st.line_chart(df_stock_[[close_col, "SMA_50", "EMA_20"]])
                else:
                    st.warning(f"⚠️ No closing price data available for {ticker}.")
            else:
                st.warning("⚠️ No valid stock data retrieved! Check API or ticker availability.")
        else:
            st.warning("⚠️ Unexpected data structure received from Alpaca API.")
  
    if stock_data:
        st.subheader(f"📊 {ticker} Stock Data from {start_date} to {end_date}")
        plot_stock_data(stock_data, ticker)
    else:
        st.warning("⚠️ No stock data retrieved. Try adjusting the date range or ticker.")

    # 📰 News Section
    news_data = fetch_news_data(ticker)
    if news_data:
        df_news = analyze_sentiment(news_data, ticker)
        if df_news is not None:
            st.subheader("📰 Latest News & Sentiment Scores")
            st.dataframe(df_news)
            display_sentiment_summary(df_news)
        else:
            st.warning("⚠️ No sentiment data available.")
    
    import pandas as pd

    # Ensure correct 'close' column
    close_col = f"{ticker}_close" if f"{ticker}_close" in df_stock_.columns else None

    if close_col:
        # ✅ Compute Moving Averages
        df_stock_["SMA_50"] = df_stock_[close_col].rolling(window=50, min_periods=1).mean()
        df_stock_["EMA_20"] = df_stock_[close_col].ewm(span=20, adjust=False).mean()

        # ✅ Convert df_news["Date"] to datetime & set as index
        df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce", utc=True)
        df_news.set_index("Date", inplace=True)

        # ✅ Merge Sentiment and Stock Data (Fixed: Ensure both indices are datetime64[ns, UTC])
        df_merged = df_stock_.merge(df_news[["Compound Score"]], left_index=True, right_index=True, how="left")

        # ✅ Generate Buy/Sell Signals
        df_merged["Buy_Signal"] = ((df_merged[close_col] > df_merged["SMA_50"]) & (df_merged["Compound Score"] > 0)).astype(int)
        df_merged["Sell_Signal"] = ((df_merged[close_col] < df_merged["SMA_50"]) & (df_merged["Compound Score"] < 0)).astype(int)

        # ✅ Debugging: Print Buy/Sell Signals to Check if they are valid
        st.write("🔍 Buy/Sell Signal Data Preview:", df_merged[["Buy_Signal", "Sell_Signal"]].tail())

        # ✅ Plot Data (Fixed: Use `df_merged` for plotting)
        st.subheader("📊 Buy/Sell Signals & Moving Averages")
        st.line_chart(df_merged[[close_col, "SMA_50", "EMA_20", "Buy_Signal", "Sell_Signal"]])
    else:
        st.warning(f"⚠️ No closing price data available for {ticker}.")
            
