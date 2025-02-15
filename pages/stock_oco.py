# stock_oco.py
from datetime import datetime
import streamlit as st
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting

# Custom CSS Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px !important;
        color: #2E86C1;
        padding-bottom: 15px;
        border-bottom: 3px solid #2E86C1;
        margin-bottom: 25px;
    }
    .strategy-card {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        margin-top: 20px;
    }
    .config-section {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 16px !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Page Header
st.markdown('<div class="main-header">📈 OCO (One-Cancels-the-Other) Strategy</div>', unsafe_allow_html=True)

# Strategy Description
st.markdown("""
    **Strategy Overview**  
    This strategy demonstrates how to use OCO (One-Cancels-the-Other) orders to manage trades:
    - **OCO Orders**: Automatically cancels one order when the other is executed
    - **Take Profit**: Sells at a specified profit target
    - **Stop Loss**: Sells at a specified loss limit
    - **Risk Management**: Protects profits and limits losses
""")
st.divider()

# OCO Strategy Class
class StockOco(Strategy):
    parameters = {
        "buy_symbol": "SPY",
        "take_profit_price": 405.0,
        "stop_loss_price": 395.0,
        "quantity": 10,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.counter = 0
        st.session_state.trade_history = []
        st.session_state.last_trade = "No trades yet"

    def on_trading_iteration(self):
        buy_symbol = self.parameters["buy_symbol"]
        take_profit_price = self.parameters["take_profit_price"]
        stop_loss_price = self.parameters["stop_loss_price"]
        quantity = self.parameters["quantity"]

        current_value = self.get_last_price(buy_symbol)
        self.log_message(f"Current {buy_symbol} price: ${current_value:.2f}")

        if self.first_iteration:
            # Market order to buy
            main_order = self.create_order(buy_symbol, quantity, "buy")
            self.submit_order(main_order)
            st.session_state.last_trade = f"🛒 Bought {quantity} shares of {buy_symbol} at market price"

            # OCO order for take profit and stop loss
            oco_order = self.create_order(
                buy_symbol,
                quantity,
                "sell",
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                type="oco",
            )
            self.submit_order(oco_order)
            st.session_state.last_trade += f"\n📊 Placed OCO order: Take Profit at ${take_profit_price:.2f}, Stop Loss at ${stop_loss_price:.2f}"

            st.session_state.trade_history.append({
                "date": self.get_datetime(),
                "action": "Initial Buy and OCO Order",
                "symbol": buy_symbol,
                "quantity": quantity,
                "take_profit": take_profit_price,
                "stop_loss": stop_loss_price,
            })

# Configuration Panel
with st.container():
    st.subheader("⚙️ Strategy Configuration")
    col1, col2 = st.columns(2)
    with col1:
        buy_symbol = st.text_input("Ticker Symbol", value="SPY")
    with col2:
        quantity = st.number_input("Quantity", min_value=1, value=10)
    
    col3, col4 = st.columns(2)
    with col3:
        take_profit_price = st.number_input("Take Profit Price", value=405.0)
    with col4:
        stop_loss_price = st.number_input("Stop Loss Price", value=395.0)
    
    live_mode = st.toggle("Live Trading Mode", value=False)

# Update Strategy Parameters
StockOco.parameters = {
    "buy_symbol": buy_symbol,
    "take_profit_price": take_profit_price,
    "stop_loss_price": stop_loss_price,
    "quantity": quantity,
}

# Backtest Configuration
if not live_mode:
    st.subheader("📅 Backtest Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 3, 3))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2023, 3, 10))

# Execution Control
if st.button("🚀 Start Analysis" if live_mode else "📊 Run Backtest"):
    if live_mode:
        st.error("Live trading is not implemented in this example.")
    else:
        with st.spinner("🔍 Running backtest..."):
            results = StockOco.backtest(
                YahooDataBacktesting,
                datetime(start_date.year, start_date.month, start_date.day),
                datetime(end_date.year, end_date.month, end_date.day),
                benchmark_asset="SPY",
            )
            
            st.subheader("📊 Performance Report")
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Strategy Return", f"{results['total_return']:.2f}%")
            with cols[1]:  
                 st.metric("Benchmark Return", f"{results.get('benchmark_return', 0):.2f}%" if isinstance(results.get('benchmark_return'), (int, float)) else "N/A")
            with cols[2]:
                st.metric("Max Drawdown", f"{float(results.get('max_drawdown', {}).get('value', 0)):.2f}%")
 
            
            if 'portfolio_value' in results:
                st.line_chart(results['portfolio_value'], use_container_width=True, color="#2E86C1")
            else:
                st.warning("Portfolio value data not available for charting.")

# Trade History
if 'trade_history' in st.session_state:
    st.subheader("📝 Trade History")
    st.dataframe(st.session_state.trade_history)

# Disclaimer
st.divider()
st.markdown("""
    <div class="disclaimer">
    *OCO orders do not guarantee execution at specified prices. Past performance is not indicative of future results. 
    Trading involves risk of loss. Backtest results are hypothetical and based on historical data.*
    </div>
    """, unsafe_allow_html=True)