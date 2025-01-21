from datetime import datetime
import streamlit as st
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import TradingFee
from lumibot.traders import Trader
from lumibot.credentials import ALPACA_CREDS
from lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from lumibot.example_strategies.stock_sentiment import StockSentiment
# Streamlit server consistently failed status checks
# Please fix the errors, push an update to the git repo, or reboot the app.


st.set_page_config(
    page_title="FinBERT: Financial Sentiment Analysis with BERT",
    layout="wide",
    page_icon=':bar_chart:',
)
st.title("Sentiment-Based Trading Bot with Live Trading")
st.subheader("FinBERT pre-trained NLP model to analyze sentiment of financial text.")
"""
Automated sentiment or polarity analysis of texts produced by financial actors using Natural Language processing (NLP) methods.
"""

import streamlit as st
import threading

def run_backtest():
    backtesting_start  = datetime(2020,1,1)
    backtesting_end   = datetime(2020,11,12) 
    st.write(f"Backtesting: Start {backtesting_start} End {backtesting_end}")
    broker      = Alpaca(ALPACA_CREDS) 
    strategy    = StockSentiment(name='mlstrat', broker=broker, 
                        parameters={"symbol":"SPY", 
                                    "cash_at_risk":.5})
    results = strategy.backtest(
        YahooDataBacktesting, 
        backtesting_start, 
        backtesting_end, 
        parameters={"symbol":"SPY", "cash_at_risk":.5}
    )
    st.write(results)
    print(results)

    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()    
    
if __name__ == "__main__":
    if st.button("Start Backtest"):
        st.write("Starting backtest in the background...")
        thread = threading.Thread(target=run_backtest)
        thread.start()
        st.write("Backtest is running...")
        


