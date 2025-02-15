# stock_bracket.py
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
    </style>
    """, unsafe_allow_html=True)

# Page Header
st.markdown('<div class="main-header">📊 Stock Bracket Order Strategy</div>', unsafe_allow_html=True)

# Strategy Description
st.markdown("""
    **Strategy Overview**  
    Automated trading strategy using bracket orders to manage risk with built-in:
    - Take profit targets
    - Stop loss protection
    - Fixed position sizing
    - Daily position evaluation
""")

st.divider()

class StockBracket(Strategy):
    parameters = {
        "buy_symbol": "SPY",
        "take_profit_price": 405.0,
        "stop_loss_price": 395.0,
        "quantity": 10,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.counter = 0

    def on_trading_iteration(self):
        buy_symbol = self.parameters["buy_symbol"]
        take_profit_price = self.parameters["take_profit_price"]
        stop_loss_price = self.parameters["stop_loss_price"]
        quantity = self.parameters["quantity"]

        current_value = self.get_last_price(buy_symbol)
        self.log_message(f"Current {buy_symbol} price: ${current_value:.2f}")

        if self.first_iteration:
            order = self.create_order(
                buy_symbol,
                quantity,
                "buy",
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                type="bracket",
            )
            self.submit_order(order)
            st.session_state.last_trade = (
                f"🔔 Bracket order placed for {quantity} {buy_symbol} shares\n"
                f"✅ Take profit: ${take_profit_price:.2f}\n"
                f"🛑 Stop loss: ${stop_loss_price:.2f}"
            )

# UI Configuration Panel
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        buy_symbol = st.text_input("Ticker Symbol", value="SPY")
    with col2:
        quantity = st.number_input("Shares Qty", min_value=1, value=10)
    with col3:
        take_profit = st.number_input("Take Profit ($)", min_value=0.01, value=405.0)
    with col4:
        stop_loss = st.number_input("Stop Loss ($)", min_value=0.01, value=395.0)

# Update strategy parameters
StockBracket.parameters = {
    "buy_symbol": buy_symbol,
    "take_profit_price": take_profit,
    "stop_loss_price": stop_loss,
    "quantity": quantity,
}

# Execution Mode Selection
st.divider()
live_mode = st.checkbox("Enable Live Trading", value=False)

if not live_mode:
    # Backtest Configuration
    st.subheader("📅 Backtest Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = (st.date_input("Start Date", datetime(2023, 3, 3)))
    with col2:
        end_date = st.date_input("End Date", datetime(2023, 3, 10))
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Running backtest..."):
            
            import datetime

            results = Strangle.backtest(
            YahooDataBacktesting,
            datetime.datetime(start_date.year, start_date.month, start_date.day),  # ✅ Correct
            datetime.datetime(end_date.year, end_date.month, end_date.day),  # ✅ Correct
            benchmark_asset="SPY",
            )
       
            st.subheader("📈 Backtest Results")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Strategy Return", f"{results['total_return']:.2f}%")
            with cols[1]:
                st.metric("Benchmark Return", f"{results['benchmark_return']:.2f}%")
            with cols[2]:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
            
            st.line_chart(results['portfolio_value'])
            st.success("Backtest completed!")
else:
    # Live Trading Configuration
    st.subheader("🔐 Live Trading Setup")
    with st.expander("Broker Credentials"):
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        ALPACA_CONFIG = {
            "API_KEY": api_key,
            "API_SECRET": api_secret,
            "PAPER": True
        }
    
    if st.button("🚀 Start Live Trading"):
        from lumibot.brokers import Alpaca
        from lumibot.traders import Trader
        
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = StockBracket(broker=broker)
        trader.add_strategy(strategy)
        
        st.session_state.trader = trader
        st.success("Live trading started!")
        
    if 'trader' in st.session_state:
        st.subheader("📢 Live Trading Updates")
        if 'last_trade' in st.session_state:
            st.code(st.session_state.last_trade)
        
        if st.button("🛑 Stop Trading"):
            st.session_state.trader.stop_all()
            del st.session_state.trader
            st.success("Trading stopped!")

# Disclaimer
st.divider()
st.markdown("""
    <div class="disclaimer">
    *Past performance is not indicative of future results. Trading involves substantial risk of loss.
    Bracket orders do not guarantee execution at specified prices. Test all strategies in backtesting mode before live trading.*
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Existing main logic remains for script execution
    is_live = False
    # ... (rest of original main code)