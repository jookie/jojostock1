import streamlit as st
import logging
import datetime
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.credentials import ALPACA_CONFIG
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

logger = logging.getLogger(__name__)

class LifecycleLogger(Strategy):
    """
    A strategy that logs key lifecycle events during a trading session.
    It does not execute trades but provides valuable insights into:
    - Market opening and closing times
    - Trading session start and end
    - Iterative trading loops
    
    Useful for debugging and understanding the trading lifecycle.
    """
    parameters = {
        "sleeptime": "10s",
        "market": "24/7",
    }

    def initialize(self, symbol=""):
        self.sleeptime = self.parameters["sleeptime"]
        self.set_market(self.parameters["market"])

    def before_market_opens(self):
        dt = self.get_datetime()
        logger.info(f"{dt} before_market_opens called")

    def before_starting_trading(self):
        dt = self.get_datetime()
        logger.info(f"{dt} before_starting_trading called")

    def on_trading_iteration(self):
        dt = self.get_datetime()
        logger.info(f"{dt} on_trading_iteration called")

    def before_market_closes(self):
        dt = self.get_datetime()
        logger.info(f"{dt} before_market_closes called")

    def after_market_closes(self):
        dt = self.get_datetime()
        logger.info(f"{dt} after_market_closes called")

# Streamlit UI
st.set_page_config(page_title="Lifecycle Logger", layout="wide")

st.markdown(
    """
    <style>
        .stButton>button {
            width: 100%;
            font-size: 18px;
            font-weight: bold;
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTitle {
            color: #1E88E5;
            font-size: 36px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='stTitle'>📊 Lifecycle Logger Strategy</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>This strategy logs key trading lifecycle events without executing trades. Use it for debugging or monitoring market behavior.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Trading Settings")
    sleeptime = st.slider("Sleep Time Between Iterations (Seconds)", 1, 60, 10)
    market_hours = st.selectbox("Market Type", ["24/7", "NYSE", "Crypto"])

with col2:
    st.subheader("Backtesting Period")
    start_date = datetime.datetime.combine(st.date_input("Start Date", datetime.date(2023, 1, 1)), datetime.datetime.min.time())
    end_date = datetime.datetime.combine(st.date_input("End Date", datetime.date(2024, 9, 1)), datetime.datetime.min.time())

st.markdown("---")

if st.button("🚀 Run Backtest", use_container_width=True):
    parameters = {
        "sleeptime": f"{sleeptime}s",
        "market": market_hours,
    }
    
    results = LifecycleLogger.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="SPY",
        parameters=parameters,
    )

    st.success("✅ Backtest completed!")
    st.markdown("### 📊 Backtest Results")
    st.write(results)

# Live Trading
if st.button("🔴 Start Live Trading", use_container_width=True):
    trader = Trader()
    broker = Alpaca(ALPACA_CONFIG)
    strategy = LifecycleLogger(broker=broker)
    trader.add_strategy(strategy)
    strategy_executors = trader.run_all()
    st.success("🔴 Live Trading Started")
