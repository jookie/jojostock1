import streamlit as st
from alpaca_trade_api.rest import REST, TimeFrame
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from lib.MLTradingBot.finbert_utils import estimate_sentiment

BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
    "API_KEY": "PKXQGLU5DJJ30MUWS2G6", 
    "API_SECRET": "vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d", 
    "PAPER": True
}
st.title("Sentiment-Based Trading Bot")

api =REST(key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"],base_url=BASE_URL)
start_date = datetime(2020, 1, 1).strftime('%Y-%m-%d')  # Correct format: '2020-01-01'
end_date = datetime(2020, 11, 1).strftime('%Y-%m-%d')  # Correct format: '2020-11-01'

try:
    news1 = api.get_news(symbol="SPY", start=start_date, end=end_date)
    print(f"news1================{news1}=============")
    st.write(f"news1================{news1}=============")
    news2 = [ev.__dict__["_raw"]["headline"] for ev in news1]
    print(f"news2================{news2}=============")
    st.write(f"news2================{news2}=============")
except Exception as e:
    print(f"Error fetching news: {e}")