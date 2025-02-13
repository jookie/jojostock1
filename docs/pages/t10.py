import streamlit as st
import threading
import time
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
import pandas as pd
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['STREAMLIT_RUNNING_IN_BARE_MODE'] = '1'

# Global progress state
class ProgressState:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()
        self.trading_days = None
        
    def initialize(self, start, end):
        """Calculate actual trading days"""
        dates = pd.bdate_range(start=start, end=end)
        self.trading_days = len(dates)
        
    def update(self, current_date):
        """Update progress based on business days"""
        with self.lock:
            days_passed = pd.bdate_range(
                start=self.start_date, 
                end=current_date
            ).nunique()
            self.value = min(days_passed / self.trading_days, 1.0)

progress = ProgressState()

class DailyProgressStrategy(Strategy):
    def initialize(self):
        self.sleeptime = "1D"  # Match Yahoo's daily data
        progress.initialize(
            self.parameters["start_date"],
            self.parameters["end_date"]
        )
        
    def on_trading_iteration(self):
        progress.update(self.get_datetime())
        time.sleep(0.01)  # Allow UI updates

def backtest_worker():
    try:
        from lumibot.credentials import ALPACA_CREDS
        broker = Alpaca(
            ALPACA_CREDS, 
            connect_stream=False
        )
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        progress.start_date = start_date
        
        backtesting = YahooDataBacktesting(
            datetime_start=start_date,
            datetime_end=end_date,
            broker=broker
        )
        
        strategy = DailyProgressStrategy(
            broker=broker,
            budget=100000,
            parameters={
                "start_date": start_date,
                "end_date": end_date
            }
        )
        
        strategy.backtest()
        progress.value = 1.0  # Force completion
        
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")

# Streamlit UI
st.title("Stock Backtest Analyzer")

if st.button("Start Analysis"):
    progress.value = 0.0
    thread = threading.Thread(target=backtest_worker, daemon=True)
    thread.start()

placeholder = st.empty()
progress_bar = st.progress(0.0)

# UI update loop
while True:
    try:
        current_value = progress.value
        progress_bar.progress(current_value)
        
        # Update every 100ms
        time.sleep(0.1)
        
        # Exit condition
        if current_value >= 0.999:
            placeholder.success("Analysis complete!")
            break
            
    except KeyboardInterrupt:
        break