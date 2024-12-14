from __future__ import annotations
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
ALPACA_CREDS = {
    "API_KEY": "PKXQGLU5DJJ30MUWS2G6", 
    "API_SECRET": "vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d", 
    "BASE_URL" : "https://paper-api.alpaca.markets",
    "PAPER": True
}
import streamlit as st
from lib.MLTradingBot.finbert_utils import estimate_sentiment
from alpaca_trade_api import REST 
# from lib.MLTradingBot.lumibot.lumibot.backtesting.yahoo_backtesting import YahooDataBacktesting
# from lib.MLTradingBot.lumibot.lumibot.strategies import Strategy
# from lib.MLTradingBot.lumibot.lumibot.traders import Trader
# from lib.MLTradingBot.lumibot.lumibot.brokers.alpaca import Alpaca
from datetime import datetime, timedelta
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api =REST(key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"])
from datetime import datetime, timedelta
from alpaca_trade_api import REST  # Adjust imports based on your setup

# Initialize API
api = REST(key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"])

# Current datetime for 'today'
today = datetime.now()

# Calculate three days prior
three_days_prior = today - timedelta(days=3)

# Format the datetime objects to RFC3339 (ISO8601) strings
start = three_days_prior.isoformat(timespec="seconds") + "Z"  # Add 'Z' to denote UTC
end = today.isoformat(timespec="seconds") + "Z"

# Fetch news
news = api.get_news(
    symbol="SPY", 
    start=start, 
    end=end
)

# Output results
st.write(today)
news = [ev.__dict__["_raw"]["headline"] for ev in news]
print(news)
probability, sentiment = estimate_sentiment(news)
st.write(f"Probability: {probability}, Setiment: {sentiment}" )
st.write(news)





