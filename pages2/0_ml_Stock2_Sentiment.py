# 📈 Stock Sentiment Analysis & Trading Strategy
from __future__ import annotations
import warnings ; warnings.filterwarnings("ignore")
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from lib.utility.util import (
    get_ticker_start_end_date,
    get_real_time_price,
    fetch_stock_data,
    fetch_news_data,
    analyze_sentiment,
    display_sentiment_summary,
    plot_stock_data
)
from alpaca_trade_api import REST
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
API_BASE_URL = 'https://paper-api.alpaca.markets'
ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY, 
    "API_SECRET": ALPACA_API_SECRET, 
    "PAPER": True
}   
from lumibot.brokers.alpaca import Alpaca    
from lumibot.backtesting import YahooDataBacktesting, PandasDataBacktesting,  PolygonDataBacktesting, AlpacaBacktesting

from lib.rl.config_tickers import index_dict
from lumibot.strategies import Strategy

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from lumibot.finbert_utils import estimate_sentiment

from lumibot.traders import Trader

st.set_page_config(page_title="Stock Sentiment & Trading", layout="wide", page_icon="📊")

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
ticker, start_date, end_date , cash_at_risk= get_ticker_start_end_date()
stock_data = fetch_stock_data(ticker, start_date, end_date)
t1 = start_date
t2 = end_date
# t1 = datetime.datetime(start_date)
# t2 = datetime.datetime(end_date  )
# t1 = datetime(2020,1,1)
# t2 = datetime(2023,12,31) 
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
    

class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=API_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)
        
    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()

        if cash > last_price: 
            if sentiment == "positive" and probability > .999: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.95
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"
    
broker = Alpaca(ALPACA_CREDS) 
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":ticker, 
                                "cash_at_risk":.5})
strategy.backtest(
    YahooDataBacktesting, 
    t1, 
    t2, 
    parameters={"symbol":ticker, "cash_at_risk":.5}
)
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()    
    
    
    