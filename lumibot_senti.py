# Integrate Streamlit for user interaction with Lumibot logic while addressing threading issues like ValueError: signal only works in main thread,I  use subprocesses to separate the Streamlit frontend from the backend logic of Lumibot and pass a symbol and place an order with alpaca and lumibot. 
from __future__ import annotations
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
BASE_URL = "https://paper-api.alpaca.markets"
API_KEY = "PKXQGLU5DJJ30MUWS2G6"
API_SECRET ="vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d"
api_secret = API_SECRET
api_key    = API_KEY
base_url   = BASE_URL
ALPACA_CREDS = {
    "API_KEY": api_key, 
    "API_SECRET": api_secret, 
    "PAPER": True
}
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
)
# import subprocess
import streamlit as st
from lib.MLTradingBot.lumibot.lumibot.backtesting.yahoo_backtesting import YahooDataBacktesting
from lib.MLTradingBot.lumibot.lumibot.strategies import Strategy
from lib.MLTradingBot.lumibot.lumibot.traders import Trader
from lib.MLTradingBot.lumibot.lumibot.brokers.alpaca import Alpaca
from lib.MLTradingBot.finbert_utils import estimate_sentiment
import datetime
from datetime import datetime 
from alpaca_trade_api import REST 
from datetime import timedelta 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


trade_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
symbol = "QQQ"
req = MarketOrderRequest(
    symbol = symbol,
    qty= 1 , # notional is specified in USD, here we specify $1.11
    side = OrderSide.BUY,
    type = OrderType.MARKET,
    time_in_force = TimeInForce.DAY,
)
res = trade_client.submit_order(req)
print(f"==========trade_client.submit_order(req) ===========:/n/n {res}")
# st.write(f"/n/n/n==========trade_client.submit_order(req) ===========: {res}")
# # get a list of orders including closed (e.g. filled) orders by specifying symbol
# req = GetOrdersRequest(
#     status = QueryOrderStatus.ALL,
#     symbols = [symbol]
# )
# orders = trade_client.get_orders(req)
# # print(f"ORDERS====={orders}")

# # get all open positions
# # ref. https://docs.alpaca.markets/reference/getallopenpositions-1
# positions = trade_client.get_all_positions()
# # print(f"======{positions}")
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







