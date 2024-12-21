# pandas/core/indexes/base.py", 
#     disallow_ndim_indexing(result)
#   File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pandas/core/indexers/utils.py", line 341, in disallow_ndim_indexing
#     raise ValueError(
# ValueError: Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. Convert to a numpy array before indexing instead.
# (base) dovpeles@Dovs-Mac-mini MLTradingBot % 
from __future__ import annotations
import sys ; sys.path.append("lib/rl")
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime 
from alpaca_trade_api import REST 
from datetime import timedelta 
from finbert_utils import estimate_sentiment

# from lumibot.backtesting.yahoo_backtesting  import YahooDataBacktesting
# from lumibot.strategies import Strategy
# from lumibot.traders import Trader
# from lumibot.brokers import Alpaca

from lib.MLTradingBot.lumibot00.lumibot.backtesting  import YahooDataBacktesting
from lib.MLTradingBot.lumibot00.lumibot.strategies import Strategy
from lib.MLTradingBot.lumibot00.lumibot.traders import Trader
from lib.MLTradingBot.lumibot00.lumibot.brokers import Alpaca

# Load environment variables from .env file
# Access API_SECRET environment variable
# api_secret = os.getenv("API_SECRET")
# api_key    = os.getenv("API_KEY")
# base_url   = os.getenv("BASE_URL")
ALPACA_CREDS = {
    "API_KEY": api_key, 
    "API_SECRET": api_secret, 
    "PAPER": True
}

class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=base_url, key_id=api_key, secret_key=api_secret)

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
        # print(news)
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

start_date  = datetime(2020,1,1)
end_date    = datetime(2020,11,12) 
# end_date    = datetime(2023,12,31) 
broker      = Alpaca(ALPACA_CREDS) 
strategy    = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.5})
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
