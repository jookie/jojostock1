# Integrate Streamlit for user interaction with Lumibot logic while addressing threading issues like ValueError: signal only works in main thread,I  use subprocesses to separate the Streamlit frontend from the backend logic of Lumibot and pass a symbol and place an order with alpaca and lumibot. 
from __future__ import annotations
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv ; 
load_dotenv(); 
api_secret = os.getenv("API_SECRET" )
api_key    = os.getenv("API_KEY")
base_url   = os.getenv("BASE_URL")
paper = True # Please do not modify this. This example is for paper trading only.
trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
)
trade_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper, url_override=trade_api_url)
symbol = "SPY"
req = MarketOrderRequest(
    symbol = symbol,
    qty= 1.1,  # notional is specified in USD, here we specify $1.11
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






