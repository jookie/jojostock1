import streamlit as st
import threading
from datetime import datetime

# LumiBot imports
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.credentials import ALPACA_CREDS
# Import your strategy (example)
from lumibot.example_strategies.stock_sentiment import StockSentiment

st.set_page_config(
    page_title="FinBERT: Financial Sentiment Analysis with BERT",
    layout="wide",
    page_icon=':bar_chart:',
)

st.title("Sentiment-Based Trading Bot with Live Trading")
st.subheader("FinBERT pre-trained NLP model to analyze sentiment of financial text.")
st.write("""
Automated sentiment or polarity analysis of texts produced by financial actors using NLP methods.
""")

# Store results in session_state so we can display them from the main thread
if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = None
if "tear_sheet_figure" not in st.session_state:
    st.session_state["tear_sheet_figure"] = None
if "backtest_running" not in st.session_state:
    st.session_state["backtest_running"] = False

def run_backtest():
    """Long-running backtest & live trading in a background thread."""
    try:
        st.session_state["backtest_running"] = True

        # Define the backtest period
        start_date = datetime(2020,1,1)
        end_date   = datetime(2020,11,12)
        print(f"[Thread] Starting backtest from {start_date} to {end_date}...")

        # Set up broker and strategy
        broker = Alpaca(ALPACA_CREDS)
        strategy = StockSentiment(
            name="mlstrat",
            broker=broker,
            parameters={
                "symbol": "SPY",
                "cash_at_risk": 0.5
            }
        )

        # Run the backtest
        results = strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            parameters={"symbol":"SPY", "cash_at_risk":0.5},
        )
        st.session_state["backtest_results"] = results

        # OPTIONAL: Generate a tear sheet if LumiBot (or you) provide a function
        # Adjust this to your actual tear sheet method:
        # e.g. `fig = results.create_tear_sheet()`
        # or   `fig = results.display_tear_sheet()`
        # Make sure the function returns a Matplotlib figure rather than printing
        # or displaying inline. If it displays inline, you need to adapt it to return a figure.

        if hasattr(results, "create_tear_sheet"):
            fig = results.create_tear_sheet()
            st.session_state["tear_sheet_figure"] = fig
        elif hasattr(results, "display_tear_sheet"):
            # If display_tear_sheet returns a figure
            fig = results.display_tear_sheet(return_fig=True)
            st.session_state["tear_sheet_figure"] = fig
        else:
            # If there's no tear sheet method, you'll need custom logic:
            print("[Thread] No tear sheet method found in results.")
            st.session_state["tear_sheet_figure"] = None

        print("[Thread] Backtest finished. Starting live trading...")

        # Start live trading (This is a blocking call)
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()  # This may never exit, be aware for hosting timeouts

    finally:
        st.session_state["backtest_running"] = False


def start_background_backtest():
    # Reset old results
    st.session_state["backtest_results"] = None
    st.session_state["tear_sheet_figure"] = None

    # Start a thread to run the backtest
    worker_thread = threading.Thread(target=run_backtest, args=())
    worker_thread.start()


# Button to trigger backtest
if st.button("Start Backtest"):
    # Prevent multiple backtest runs
    if not st.session_state["backtest_running"]:
        st.write("Backtest is starting in the background...")
        start_background_backtest()
    else:
        st.warning("A backtest is already running!")

# Status / Results display
if st.session_state["backtest_running"]:
    st.info("Backtest is running... (this might take a while).")

elif st.session_state["backtest_results"] is not None:
    st.success("Backtest completed!")
    st.write("### Backtest Results")
    st.write(st.session_state["backtest_results"])

    # Display the tear sheet if we have it
    if st.session_state["tear_sheet_figure"] is not None:
        st.write("### Tear Sheet")
        st.pyplot(st.session_state["tear_sheet_figure"])
    else:
        st.warning("No tear sheet figure available.")
