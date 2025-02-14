import streamlit as st
import datetime
from lumibot.brokers import Ccxt
from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

class ImportantFunctions(Strategy):
    """
    A cryptocurrency trading strategy that:
    - Places market and limit orders for BTC
    - Fetches and analyzes historical price data
    - Uses technical indicators like RSI, MACD, and EMA
    - Monitors positions, orders, and portfolio value
    """
    
    def initialize(self):
        self.sleeptime = "30S"
        self.set_market("24/7")
    
    def on_trading_iteration(self):
        base = Asset(symbol="BTC", asset_type="crypto")
        quote = self.quote_asset
        
        # Placing Orders
        mkt_order = self.create_order(base, 0.1, "buy", quote=quote)
        self.submit_order(mkt_order)
        lmt_order = self.create_order(base, 0.1, "buy", quote=quote, limit_price=10000)
        self.submit_order(lmt_order)
        
        # Fetching Historical Data
        bars = self.get_historical_prices(base, 100, "minute", quote=quote)
        if bars is not None:
            df = bars.df
            max_price = df["close"].max()
            self.log_message(f"Max price for {base} was {max_price}")
            
            # Technical Analysis
            rsi = df.ta.rsi(length=20).iloc[-1]
            macd = df.ta.macd().iloc[-1]
            ema = df.ta.ema(length=55).iloc[-1]
            self.log_message(f"RSI: {rsi}, MACD: {macd}, EMA: {ema}")
        
        # Checking Portfolio and Orders
        positions = self.get_positions()
        for position in positions:
            self.log_message(f"Position: {position}")
        
        portfolio_value = self.portfolio_value
        cash = self.cash
        self.log_message(f"Portfolio Value: {portfolio_value}, Cash: {cash}")

# Streamlit UI
st.set_page_config(page_title="Crypto Trading Dashboard", layout="wide")

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

st.markdown("<div class='stTitle'>📈 Crypto Trading Strategy</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Automated cryptocurrency trading using market orders, technical analysis, and portfolio monitoring.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Trading Parameters")
    api_key = st.text_input("API Key", type="password")
    secret_key = st.text_input("Secret Key", type="password")
    trade_quantity = st.slider("Trade Quantity (BTC)", 0.01, 1.0, 0.1)

with col2:
    st.subheader("Market & Analysis")
    start_date = datetime.datetime.combine(st.date_input("Start Date", datetime.date(2023, 1, 1)), datetime.datetime.min.time())
    end_date = datetime.datetime.combine(st.date_input("End Date", datetime.date(2024, 1, 1)), datetime.datetime.min.time())

st.markdown("---")

if st.button("🚀 Start Trading", use_container_width=True):
    if not api_key or not secret_key:
        st.error("❌ Please enter your API credentials.")
    else:
        KRAKEN_CONFIG = {
            "exchange_id": "kraken",
            "apiKey": api_key,
            "secret": secret_key,
            "margin": True,
            "sandbox": False,
        }
        broker = Ccxt(KRAKEN_CONFIG)
        strategy = ImportantFunctions(broker=broker)
        trader = Trader()
        trader.add_strategy(strategy)
        strategy_executors = trader.run_all()
        st.success("✅ Trading Started!")
