# run_backtest.py
import math
import sys
# sys.path.append('/Users/dovpeles/dov/jojostock1/lib/MLTradingBot')

from datetime import datetime 
import streamlit as st
from lib.MLTradingBot.lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from lib.MLTradingBot.lumibot.example_strategies.stock_bracket import StockBracket
if __name__ == "__main__":
    # Import Momentum
    
    # Backtest this strategy
    backtesting_start = datetime(2023, 1, 1)
    backtesting_end = datetime(2024, 8, 1)
    
    results = StockBracket.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )    
    st.write(f"start: {backtesting_start} end: backtesting_end {backtesting_end}")
    st.write(results)
