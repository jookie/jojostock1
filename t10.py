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

if __name__ == "__main__":
    backtesting_start = datetime(2023, 1, 2)
    backtesting_end  = datetime(2024, 10, 31)

    st.write(f"Backtesting: Start {backtesting_start} End {backtesting_end}")
    IS_BACKTESTING = True
    if not IS_BACKTESTING:
        print("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")
        st.write("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")        
        exit()
    else:
        parameters = {
            "market": "NYSE",
            "sleeptime": "1D",
            "drift_threshold": "0.05",
            "acceptable_slippage": "0.005",  # 50 BPS
            "fill_sleeptime": 15,
            "target_weights": {
                "SPY": "0.60",
                "TLT": "0.40"
            },
            "shorting": False
        }




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
        backtesting_start = datetime(2023, 6, 1)
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
        # print("Backtest result: ", result)
        st.write(f"Backtest result: {result}")
