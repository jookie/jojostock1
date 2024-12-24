
from datetime import datetime , timedelta 
import streamlit as st
from lumibot.entities import Asset, Order
from lumibot.backtesting import CcxtBacktesting
from lumibot.credentials import IS_BACKTESTING
from lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import TradingFee
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.example_strategies.stock_diversified_leverage import DiversifiedLeverage

"""
Strategy Description

This strategy will buy a few symbols that have 2x or 3x returns (have leverage), but will 
also diversify and rebalance the portfolio often.
"""

if __name__ == "__main__":
    is_live = False

    if is_live:
        ####
        # Run the strategy live
        ####
        from lumibot.credentials import ALPACA_CREDS

        trader = Trader()
        broker = Alpaca(ALPACA_CREDS)
        strategy = DiversifiedLeverage(broker=broker)
        trader.add_strategy(strategy)
        trader.run_all()

    else:
        ####
        # Backtest the strategy
        ####

        # Choose the time from and to which you want to backtest
        backtesting_start = datetime(2010, 6, 1)
        backtesting_end = datetime(2023, 7, 31)

        # 0.01% trading/slippage fee
        trading_fee = TradingFee(percent_fee=0.005)

        # Initialize the backtesting object
        print("Starting Backtest...")
        result = DiversifiedLeverage.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            benchmark_asset="SPY",
            parameters={},
            buy_trading_fees=[trading_fee],
            sell_trading_fees=[trading_fee],
        )

        print("Backtest result: ", result)
        st.write(f"Backtest result: {result}")
