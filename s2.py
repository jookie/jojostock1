from __future__ import annotations
# import streamlit as st
import warnings ; warnings.filterwarnings("ignore")

ALPACA_CREDS = {
    "API_KEY": "PKXQGLU5DJJ30MUWS2G6", 
    "API_SECRET": "vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d", 
    "BASE_URL" : "https://paper-api.alpaca.markets",
    "PAPER": True
}
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
)

from lib.MLTradingBot.lumibot.backtesting.yahoo_backtesting import YahooDataBacktesting
from lib.MLTradingBot.lumibot.strategies import Strategy
from lib.MLTradingBot.lumibot.traders import Trader
from lib.MLTradingBot.lumibot.brokers.alpaca import Alpaca
from lib.MLTradingBot.finbert_utils import estimate_sentiment
import datetime
from datetime import datetime 
from alpaca_trade_api import REST 
from datetime import timedelta 
trade_client = TradingClient(api_key=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"], paper=True)
symbol = "SPY"
req = MarketOrderRequest(
    symbol = symbol,
    qty= 1 ,
    side = OrderSide.BUY,
    type = OrderType.MARKET,
    time_in_force = TimeInForce.DAY,
)
res = trade_client.submit_order(req)
print(f"==========res = trade_client.submit_order(req)===========: {res}")
