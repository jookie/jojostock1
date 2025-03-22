
import streamlit as st
from alpaca_trade_api.rest import REST, TimeFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://paper-api.alpaca.markets"  # Use for paper trading
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
# Initialize Alpaca client
alpaca = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url=BASE_URL)

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit app setup
st.title("Sentiment-Based Trading Bot")
st.write("Analyze sentiment and trade stocks using Alpaca API.")

# User inputs
symbol = st.text_input("Stock Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
end_date = st.date_input("End Date", value=datetime.now())
sentiment_threshold = st.slider("Sentiment Threshold", -1.0, 1.0, 0.0)

# Fetch historical price data
if st.button("Fetch Data"):
    st.write(f"Fetching historical data for {symbol}...")
    bars = alpaca.get_bars(symbol, TimeFrame.Day, start_date, end_date).df
    st.write(bars.tail())

# Sentiment analysis
if st.button("Analyze Sentiment"):
    st.write("Performing sentiment analysis...")
    data = {
        "date": pd.date_range(start=start_date, end=end_date),
        "news": [f"Sample headline for {symbol} on day {i}" for i in range((end_date - start_date).days)],
    }
    
# ✅ Collect data arrays safely
dates, opens, closes, highs, lows, volumes = [], [], [], [], [], []

for bar in bars:
    dates.append(bar.timestamp)
    opens.append(bar.open)
    closes.append(bar.close)
    highs.append(bar.high)
    lows.append(bar.low)
    volumes.append(bar.volume)

# ✅ Align all arrays
min_len = min(len(dates), len(opens), len(closes), len(highs), len(lows), len(volumes))
data = {
    'Date': dates[:min_len],
    'Open': opens[:min_len],
    'Close': closes[:min_len],
    'High': highs[:min_len],
    'Low': lows[:min_len],
    'Volume': volumes[:min_len],
}
df = pd.DataFrame(data)

df["sentiment"] = df["news"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
st.write(df)
st.line_chart(df["sentiment"])

# Trading logic
if st.button("Execute Trades"):
    st.write("Executing trades...")
    for index, row in df.iterrows():
        if row["sentiment"] > sentiment_threshold:
            st.write(f"BUY: {symbol} on {row['date']} (Sentiment: {row['sentiment']})")
            alpaca.submit_order(symbol, qty=1, side="buy", type="market", time_in_force="gtc")
        elif row["sentiment"] < -sentiment_threshold:
            st.write(f"SELL: {symbol} on {row['date']} (Sentiment: {row['sentiment']})")
            alpaca.submit_order(symbol, qty=1, side="sell", type="market", time_in_force="gtc")

st.write("Adjust parameters and try again!")