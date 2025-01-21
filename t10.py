import streamlit as st
import threading
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.credentials import ALPACA_CREDS
from lumibot.example_strategies.stock_sentiment import StockSentiment

st.set_page_config(
    page_title="FinBERT: Financial Sentiment Analysis with BERT",
    layout="wide",
    page_icon=':bar_chart:',
)

st.title("Sentiment-Based Trading Bot with Live Trading")
st.subheader("FinBERT pre-trained NLP model to analyze sentiment of financial text.")
st.write("""
Automated sentiment or polarity analysis of texts produced by financial actors using Natural Language Processing (NLP) methods.
""")

# A global or session_state variable to store results
if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = None
if "thread" not in st.session_state:
    st.session_state["thread"] = None

def run_backtest():
    backtesting_start = datetime(2020,1,1)
    backtesting_end   = datetime(2020,11,12)
    print(f"Backtesting from {backtesting_start} to {backtesting_end} ...")

    broker = Alpaca(ALPACA_CREDS)
    strategy = StockSentiment(
        name="mlstrat",
        broker=broker,
        parameters={
            "symbol": "SPY",
            "cash_at_risk": 0.5
        }
    )

    # Perform the backtest (avoid st.write in background thread)
    results = strategy.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        parameters={"symbol": "SPY", "cash_at_risk": 0.5}
    )
    # Store the results for the main thread to display
    st.session_state["backtest_results"] = results

    # Optionally start live trading
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()

def start_backtest_in_thread():
    # Reset previous results
    st.session_state["backtest_results"] = None

    # Create and start thread
    t = threading.Thread(target=run_backtest, args=())
    t.start()
    st.session_state["thread"] = t

# Button to start the backtest in a thread
if st.button("Start Backtest"):
    if st.session_state["thread"] is None or not st.session_state["thread"].is_alive():
        st.write("Starting backtest in background...")
        start_backtest_in_thread()
    else:
        st.write("A backtest is already running.")

# Display results if available
if st.session_state["backtest_results"] is not None:
    st.success("Backtest completed!")
    st.write(st.session_state["backtest_results"])
