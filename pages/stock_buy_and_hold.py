# stock_buy_and_hold.py
from datetime import datetime
import streamlit as st
from lumibot.strategies.strategy import Strategy

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
    .metric-box {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    </style>
    """, unsafe_allow_html=True)

# Page Header
st.markdown('<div class="main-header">📈 Buy & Hold Strategy</div>', unsafe_allow_html=True)

# Strategy Description
st.markdown("""
    **Strategy Overview**  
    Classic buy-and-hold investment approach:
    - Invests entire portfolio in selected asset
    - No active trading after initial purchase
    - Automatically adjusts position size based on portfolio value
    - Daily portfolio valuation updates
""")
st.divider()

class BuyAndHold(Strategy):
    parameters = {
        "buy_symbol": "QQQ",
    }

    def initialize(self):
        self.sleeptime = "1D"
        st.session_state.portfolio_history = []
        st.session_state.last_trade = "No trades yet"

    def on_trading_iteration(self):
        buy_symbol = self.parameters["buy_symbol"]
        dt = self.get_datetime()
        current_value = self.get_last_price(buy_symbol)
        
        # Update portfolio history
        portfolio_value = self.portfolio_value
        st.session_state.portfolio_history.append({
            "date": dt,
            "value": portfolio_value,
            "asset_price": current_value
        })
        
        all_positions = self.get_positions()
        if len(all_positions) <= 1:
            quantity = int(self.portfolio_value // current_value)
            if quantity > 0:
                purchase_order = self.create_order(buy_symbol, quantity, "buy")
                self.submit_order(purchase_order)
                st.session_state.last_trade = (
                    f"🛒 Purchased {quantity} shares of {buy_symbol}\n"
                    f"💰 Total Investment: ${quantity * current_value:,.2f}\n"
                    f"📅 {dt.strftime('%Y-%m-%d %H:%M:%S')}"
                )

# Configuration Panel
with st.container():
    st.subheader("⚙️ Strategy Configuration")
    col1, col2 = st.columns([2, 3])
    with col1:
        symbol = st.text_input("Asset Ticker", value="QQQ")
    with col2:
        live_mode = st.toggle("Live Trading Mode", value=False)

# Backtest Configuration
if not live_mode:
    st.subheader("📅 Backtest Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2024, 9, 1))

# Live Trading Setup
else:
    st.subheader("🔐 Broker Configuration")
    with st.expander("Credentials Setup"):
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        ALPACA_CONFIG = {
            "API_KEY": api_key,
            "API_SECRET": api_secret,
            "PAPER": True
        }

# Execution Control
if st.button("🚀 Start Analysis" if live_mode else "📊 Run Backtest"):
    BuyAndHold.parameters["buy_symbol"] = symbol
    
    if live_mode:
        from lumibot.brokers import Alpaca
        from lumibot.traders import Trader
        
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = BuyAndHold(broker=broker)
        trader.add_strategy(strategy)
        
        st.session_state.trader = trader
        st.success("Live trading initialized!")
    else:
        from lumibot.backtesting import YahooDataBacktesting
        
        with st.spinner("🔍 Running backtest..."):
            results = BuyAndHold.backtest(
                YahooDataBacktesting,
                datetime(start_date.year, start_date.month, start_date.day),
                datetime(end_date.year, end_date.month, end_date.day),
                benchmark_asset="SPY",
            )
            
            st.subheader("📊 Performance Report")
            
            # Metrics   DOV
            cols = st.columns(3)
            with cols[0]:
                st.metric("Strategy Return", f"{results['total_return']:.2f}%")
            with cols[1]:  
                 st.metric("Benchmark Return", f"{results.get('benchmark_return', 0):.2f}%" if isinstance(results.get('benchmark_return'), (int, float)) else "N/A")
            with cols[2]:
                st.metric("Max Drawdown", f"{float(results.get('max_drawdown', {}).get('value', 0)):.2f}%")
 
            import pandas as pd

            # Retrieve portfolio value safely
            portfolio_data = results.get("portfolio_value", [])

            # Ensure it's a valid DataFrame or Series
            if isinstance(portfolio_data, (list, dict)):
                portfolio_df = pd.DataFrame(portfolio_data)

                if not portfolio_df.empty and len(portfolio_df.columns) > 0:
                    st.line_chart(portfolio_df, use_container_width=True)  # Remove explicit color
                else:
                    st.warning("⚠️ No portfolio data available for visualization.")
            else:
                st.warning("⚠️ No portfolio data found in results.")

            # Portfolio Value Chart
            # st.line_chart(results.get("portfolio_value", []), use_container_width=True, color="#2E86C1")
            # DOV
            # st.line_chart(
                
            #     data=results['portfolio_value'], 
            #     use_container_width=True,
            #     color="#2E86C1"
            # )
            
            # Trade History
            with st.expander("📝 Detailed Trade Log"):
                st.write(results.get("trade_history", "⚠️ No trade history available."))
                # st.write(results['trade_history'])

# Live Trading Updates
if live_mode and 'trader' in st.session_state:
    st.subheader("📢 Live Portfolio Updates")
    
    if 'portfolio_history' in st.session_state:
        latest = st.session_state.portfolio_history[-1]
        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Value", f"${latest['value']:,.2f}")
        with cols[1]:
            st.metric("Asset Price", f"${latest['asset_price']:,.2f}")
        with cols[2]:
            st.metric("Last Update", latest['date'].strftime("%Y-%m-%d %H:%M"))
    
    st.code(st.session_state.last_trade)
    
    if st.button("🛑 Stop Trading"):
        st.session_state.trader.stop_all()
        del st.session_state.trader
        st.success("Trading session stopped!")

# Disclaimer
st.divider()
st.markdown("""
    <div class="disclaimer">
    *Past performance is not indicative of future results. Buy-and-hold strategy does not protect against market declines. 
    Investment involves risk of loss. Backtest results are hypothetical and based on historical data.*
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    IS_BACKTESTING = True
    # ... (original main code remains for script execution)