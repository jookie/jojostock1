# simple_start_single_file.py
from datetime import datetime
import streamlit as st
from lumibot.credentials import ALPACA_CREDS
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 32px !important;
        color: #2E86C1;
        padding-bottom: 15px;
        border-bottom: 2px solid #2E86C1;
    }
    .strategy-card {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .metric-box {
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

class MyStrategy(Strategy):
    def initialize(self, symbol="", quantity=1, sleeptime=180):
        self.sleeptime = sleeptime
        self.symbol = symbol
        self.quantity = quantity
        self.side = "buy"

    def on_trading_iteration(self):
        if self.get_position(self.symbol) is None:
            order = self.create_order(self.symbol, self.quantity, self.side)
            self.submit_order(order)
            st.session_state.last_trade = f"Bought {self.quantity} shares of {self.symbol}"
        else:
            st.session_state.last_trade = f"Already holding {self.symbol}, skipping trade"

if __name__ == "__main__":
    # st.set_page_config(page_title="Trading Bot Interface", layout="wide")
    
    # UI Header
    st.markdown('<div class="main-header">🚀 Automated Trading Bot</div>', unsafe_allow_html=True)
    
    # Strategy Description
    with st.expander("📖 Strategy Overview", expanded=True):
        st.markdown("""
        **Simple Momentum Strategy**:
        - Buys specified asset at regular intervals
        - Basic position management (no duplicate positions)
        - Configurable parameters:
          - Trade interval (minutes)
          - Position size
          - Target symbol
        """)
    
    # Configuration Panel
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("📈 Trading Symbol", value="SPY")
        with col2:
            quantity = st.number_input("🔢 Position Size", min_value=1, value=1)
        with col3:
            sleeptime = st.number_input("⏱ Trade Interval (minutes)", min_value=1, value=180)
        
        live_trading = st.checkbox("🔴 Live Trading Mode", value=True)
     
    # Backtest Parameters
    if not live_trading:
        st.subheader("📅 Backtest Configuration")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime(2020, 12, 31))
    
    # Execution Control
    if st.button("🚀 Start Trading" if live_trading else "📊 Run Backtest"):
        trader = Trader()
        broker = Alpaca(ALPACA_CREDS) if live_trading else None
        
            
        strategy = MyStrategy(
            broker=broker,
            symbol=symbol,
            quantity=quantity,
            sleeptime=sleeptime
        )
        
        if live_trading:
            with st.status("📡 Connecting to Live Markets..."):
                trader.add_strategy(strategy)
                st.session_state.trader = trader
                st.rerun()
        else:
            with st.spinner("⏳ Running Backtest..."):
                results = strategy.backtest(
                    YahooDataBacktesting,
                    datetime(start_date.year, start_date.month, start_date.day),
                    datetime(end_date.year, end_date.month, end_date.day),
                    symbol=symbol
                )
                
                st.subheader("📊 Backtest Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.2f}%")
                with col2:
                    st.metric("Benchmark Return", f"{results['benchmark_return']:.2f}%")
                with col3:
                    st.metric("Win Rate", f"{results['win_rate']:.2f}%")
                
                st.line_chart(results['portfolio_value'])
    
    # Live Trading Updates
    if live_trading and 'trader' in st.session_state:
        st.subheader("📰 Live Trading Updates")
        if 'last_trade' in st.session_state:
            st.markdown(f"```{st.session_state.last_trade}```")
        
        if st.button("🛑 Stop Trading"):
            st.session_state.trader.stop_all()
            st.success("Trading stopped successfully")
            del st.session_state.trader
    
    # Disclaimer
    st.divider()
    st.caption("⚠️ Disclaimer: Trading involves risk. Past performance is not indicative of future results.")