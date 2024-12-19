
from __future__ import annotations
# import sys ; sys.path.append("~/lib/MLTradingbot")
# sys.path.append("../lib")
# import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
import os
from datetime import datetime 
from alpaca_trade_api.rest import REST 
from lib.sentiment.finbert_utils import estimate_sentiment

from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.brokers.alpaca import Alpaca
from lumibot.backtesting.yahoo_backtesting   import YahooDataBacktesting

os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_URL = "https://paper-api.alpaca.markets"
from timedelta import Timedelta
ALPACA_CREDS = {
    "API_KEY": "PKXQGLU5DJJ30MUWS2G6", 
    "API_SECRET": "vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d", 
    "PAPER": True
}
class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"])

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        print(probability, sentiment , news)
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
# ====DOV  MODICATION=======
start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31) 
# start_date = datetime(2023,1,1)
# end_date = datetime(2023,1,2) 

broker = Alpaca(ALPACA_CREDS) 

strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.5})
strategy.backtest(
    YahooDataBacktesting, c
    start_date, 
    end_date, 
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
