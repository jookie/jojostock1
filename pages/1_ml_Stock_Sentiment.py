import nltk
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from alpaca.data.models.bars import BarSet
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from lib.rl.meta.data_processor import DataProcessor
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET

# Download NLTK dependencies
nltk.download("vader_lexicon")

# Custom CSS for Streamlit UI
st.markdown(
    """
    <style>
        body { background-color: black; font-family: "Times New Roman", Times, serif; color: white; line-height: 1.6; }
        h1 { color: #3498db; }
        h2 { color: #e74c3c; }
        p { margin: 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define available tickers
FINVIZ_URL = "https://finviz.com/quote.ashx?t="
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "NFLX", "FB", "BRK.B", "V",
    "NVDA", "DIS", "BA", "IBM", "GE", "PG", "JNJ", "KO", "MCD", "T",
    "ADBE", "CRM", "INTC", "ORCL", "HD"
]

def fetch_alpaca_history(api_key, api_secret, ticker):
    """Fetch historical stock data from Alpaca."""
    client = StockHistoricalDataClient(api_key, api_secret)
    request_params = StockBarsRequest(
        symbol_or_symbols=[ticker], timeframe=TimeFrame.Day,
        start=datetime.datetime(2025, 1, 1), end=datetime.datetime(2025, 3, 16)
    )
    return client.get_stock_bars(request_params)

def fetch_news(ticker):
    """Retrieve news headlines from Finviz."""
    req = Request(url=FINVIZ_URL + ticker, headers={"user-agent": "my-app"})
    html = BeautifulSoup(urlopen(req), features="html.parser")
    return html.find(id="news-table")

def analyze_sentiment(news_table, ticker):
    """Analyze sentiment scores of news headlines."""
    parsed_data = []
    for row in news_table.findAll("tr"):
        if row.a:
            title = row.a.text
            date_data = row.td.text.split()
            date, time = (date_data[1], date_data[0]) if len(date_data) > 1 else ("Unknown", "")
            parsed_data.append([ticker, date, time, title])

    df = pd.DataFrame(parsed_data, columns=["Ticker", "Date", "Time", "Headline"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Compound Score"] = df["Headline"].apply(lambda title: SentimentIntensityAnalyzer().polarity_scores(title)["compound"])
    
    return df

def display_sentiment_summary(df):
    """Display sentiment summary statistics."""
    summary = {
        "Average Score": df["Compound Score"].mean(),
        "Positive": (df["Compound Score"] > 0).sum() / len(df) * 100,
        "Negative": (df["Compound Score"] < 0).sum() / len(df) * 100,
        "Neutral": (df["Compound Score"] == 0).sum() / len(df) * 100,
    }
    st.subheader("Sentiment Summary")
    st.write(summary)

def plot_stock_data(data, ticker):
    """Plot stock price movements."""
    if isinstance(data, BarSet):
        df_list = []
        for symbol in data.data.keys():
            bars = data[symbol]
            if isinstance(bars, list) and bars:
                df = pd.DataFrame([bar.dict() for bar in bars])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df = df.add_prefix(f"{symbol}_")
                df_list.append(df)
        
        if df_list:
            combined_data = pd.concat(df_list, axis=1).loc[:, ~pd.concat(df_list, axis=1).columns.duplicated()]
            st.write(combined_data.head())
            
            close_col = f"{ticker}_close" if f"{ticker}_close" in combined_data.columns else None
            if close_col:
                plt.figure(figsize=(10, 6))
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

def main():
    """Main Streamlit app function."""
    ticker = st.selectbox("Select a stock ticker symbol or enter your own:", TICKERS)
    if ticker:
        stock_data = fetch_alpaca_history(ALPACA_API_KEY, ALPACA_API_SECRET, ticker)
        plot_stock_data(stock_data, ticker)
        
        news_table = fetch_news(ticker)
        if news_table:
            df = analyze_sentiment(news_table, ticker)
            st.subheader("News Headlines and Sentiment Scores")
            st.dataframe(df)
            display_sentiment_summary(df)
        else:
            st.write("No news found for the entered stock ticker symbol.")

if __name__ == "__main__":
    main()
