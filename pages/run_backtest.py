# run_backtest.py
import sys
sys.path.append('/Users/dovpeles/dov/jojostock1/lib/MLTradingBot')

from datetime import datetime , timedelta 
import streamlit as st
from lumibot.backtesting import YahooDataBacktesting

if __name__ == "__main__":
    # from stock_momentum import Momentum
    from  lumibot.example_strategies.stock_bracket import StockBracket
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
