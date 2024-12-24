from datetime import datetime
import streamlit as st
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import TradingFee
from lumibot.traders import Trader
from lumibot.example_strategies.stock_diversified_leverage import DiversifiedLeverage
from lumibot.credentials import IS_BACKTESTING
from lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from lumibot.example_strategies.stock_sentiment import StockSentiment
from lumibot.credentials import ALPACA_CREDS
"""
Strategy Description
This strategy will buy a few symbols that have 2x or 3x returns (have leverage), but will 
also diversify and rebalance the portfolio often.
"""
if __name__ == "__main__":
    
    
    print("==================MLTrader(Strategy)=============================")

    
    start_date = datetime(2020,1,1)
    end_date = datetime(2023,12,31) 
    broker = Alpaca(ALPACA_CREDS) 
    strategy = StockSentiment(name='mlstrat', broker=broker, 
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
    