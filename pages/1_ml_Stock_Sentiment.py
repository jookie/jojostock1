# Stock Sentiment Analysis & Trading Strategy
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

# Download NLTK dependencies
nltk.download("vader_lexicon")

# UI Configuration
st.set_page_config(layout="wide")
st.title("📈 Stock Sentiment Analysis & Trading Strategy")
st.markdown("Analyze stock market trends using sentiment analysis and historical data.")

# Define available tickers
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "NFLX", "FB", "BRK.B", "V",
    "NVDA", "DIS", "BA", "IBM", "GE", "PG", "JNJ", "KO", "MCD", "T",
    "ADBE", "CRM", "INTC", "ORCL", "HD"
]
FINVIZ_URL = "https://finviz.com/quote.ashx?t="

# UI Layout: Ticker & Date Selection in a Single Row
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker = st.selectbox("Select a Stock Ticker:", TICKERS)

with col2:
    start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))

with col3:
    end_date = st.date_input("End Date", datetime.date.today())

# Cached API Client
@st.cache_resource
def get_stock_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# Fetch Stock Data from Alpaca
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

# Fetch News Data from Finviz
@st.cache_data
def fetch_news_data(ticker):
    """Retrieve news headlines from Finviz and return a serializable format."""
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

            return news_list  # Return as a list of dictionaries (serializable)

    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return None

# Analyze Sentiment from News Headlines
@st.cache_data
def analyze_sentiment(news_list, ticker):
    """Analyze sentiment scores of news headlines from a list of dictionaries."""
    if not news_list:
        return None

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(news_list)

    # Ensure Date column is properly formatted
    df["Date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Perform sentiment analysis
    df["Compound Score"] = df["title"].apply(lambda title: SentimentIntensityAnalyzer().polarity_scores(title)["compound"])
    
    return df


# Display Sentiment Summary
def display_sentiment_summary(df):
    summary = {
        "Average Score": df["Compound Score"].mean(),
        "Positive": (df["Compound Score"] > 0).sum() / len(df) * 100,
        "Negative": (df["Compound Score"] < 0).sum() / len(df) * 100,
        "Neutral": (df["Compound Score"] == 0).sum() / len(df) * 100,
    }
    st.subheader("Sentiment Summary")
    st.write(summary)

# Plot Stock Data
def plot_stock_data(data, ticker):
    if isinstance(data, BarSet):
        df_list = []
        for symbol, bars in data.data.items():
            if isinstance(bars, list) and bars:
                df = pd.DataFrame([bar.dict() for bar in bars])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df.add_prefix(f"{symbol}_")
                df_list.append(df)
        
        if df_list:
            combined_data = pd.concat(df_list, axis=1).loc[:, ~pd.concat(df_list, axis=1).columns.duplicated()]
            close_col = f"{ticker}_close" if f"{ticker}_close" in combined_data.columns else None
            
            if close_col:
                st.line_chart(combined_data[close_col])
            else:
                st.warning(f"⚠️ No closing price data available for {ticker}.")
        else:
            st.warning("⚠️ No valid stock data retrieved! Check API or ticker availability.")
    else:
        st.warning("⚠️ Unexpected data structure received from Alpaca API.")

# Main App Logic
if ticker:
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    if stock_data:
        st.subheader(f"{ticker} Stock Data from {start_date} to {end_date}")
        plot_stock_data(stock_data, ticker)
    else:
        st.warning("⚠️ No stock data retrieved. Try adjusting the date range or ticker.")

    news_table = fetch_news_data(ticker)
    if news_table:
        df = analyze_sentiment(news_table, ticker)
        if df is not None:
            st.subheader("📰 News Headlines & Sentiment Scores")
            st.dataframe(df)
            display_sentiment_summary(df)
        else:
            st.warning("⚠️ No sentiment data available.")
    else:
        st.warning("⚠️ No news found for the entered stock ticker symbol.")
