
from __future__ import annotations
import sys ; sys.path.append("~/lib/MLTradingbot")
sys.path.append("../lib")
# import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime 
from lib.utility.jprint import jprint
from lib.rl.config_tickers import DOW_30_TICKER
from lib.rl.meta.preprocessor.yahoodownloader import YahooDownloader
from lib.rl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from lib.rl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.rl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from lib.rl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from alpaca_trade_api import REST 

from lib.lumibot.lumibot.brokers import Alpaca
from lib.MLTradingBot.finbert_utils import estimate_sentiment
from lib.lumibot.lumibot.backtesting import YahooDataBacktesting
from lib.lumibot.lumibot.strategies.strategy import Strategy
from lib.lumibot.lumibot.traders import Trader

API_KEY = "PKEJH4W0URAU56SHKQW3" 
API_SECRET = "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY": API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}
ALPACA_CONFIG = {
    "API_KEY":  "PKEJH4W0URAU56SHKQW3" ,
    "API_SECRET": "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow",
    "PAPER": True
}

class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

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

start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31) 

broker = Alpaca(ALPACA_CREDS) 
# broker = Alpaca(api_key=ALPACA_CREDS['API_KEY'], secret_key=ALPACA_CREDS['API_SECRET'])





strategy = MLTrader(name='mlstrat', broker=broker, 
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
