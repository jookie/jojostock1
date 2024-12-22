import sys
from datetime import datetime , timedelta 
import streamlit as st
from lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from lumibot.example_strategies.stock_bracket import StockBracket
if __name__ == "__main__":
    import signal
    import streamlit as st
    backtesting_start = datetime(2023, 1, 1)
    backtesting_end = datetime(2024, 8, 1)
    st.write(f"backtesting_start {backtesting_start}")
    results = StockBracket.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )    
    st.write(results)
