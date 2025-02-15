# stock_limit_and_trailing_stops.py
from datetime import datetime
import streamlit as st
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
# update to add UI , styles, add UI , styles ,header and the description below the header
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

# Strategy Class
class LimitAndTrailingStop(Strategy):
    parameters = {
        "buy_symbol": "SPY",
        "limit_buy_price": 403.0,
        "limit_sell_price": 407.0,
        "trail_percent": 0.02,
        "trail_price": 7.0,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.counter = 0
        st.session_state.trade_history = []
        st.session_state.last_trade = "No trades yet"

    def on_trading_iteration(self):
        buy_symbol = self.parameters["buy_symbol"]
        limit_buy_price = self.parameters["limit_buy_price"]
        limit_sell_price = self.parameters["limit_sell_price"]
        trail_percent = self.parameters["trail_percent"]
        trail_price = self.parameters["trail_price"]

        current_value = self.get_last_price(buy_symbol)
        self.log_message(f"Current {buy_symbol} price: ${current_value:.2f}")

        if self.first_iteration:
            # Place limit buy order
            purchase_order = self.create_order(buy_symbol, 100, "buy", limit_price=limit_buy_price)
            self.submit_order(purchase_order)
            st.session_state.last_trade = f"🛒 Placed limit buy order for {buy_symbol} at ${limit_buy_price:.2f}"

            # Place limit sell order
            sell_order = self.create_order(buy_symbol, 100, "sell", limit_price=limit_sell_price)
            self.submit_order(sell_order)
            st.session_state.last_trade += f"\n💰 Placed limit sell order at ${limit_sell_price:.2f}"

            # Place trailing stop orders
            trailing_pct_stop_order = self.create_order(buy_symbol, 100, "sell", trail_percent=trail_percent)
            self.submit_order(trailing_pct_stop_order)
            st.session_state.last_trade += f"\n📉 Placed trailing stop ({trail_percent * 100:.2f}%)"

            trailing_price_stop_order = self.create_order(buy_symbol, 50, "sell", trail_price=trail_price)
            self.submit_order(trailing_price_stop_order)
            st.session_state.last_trade += f"\n📉 Placed trailing stop (${trail_price:.2f})"

            st.session_state.trade_history.append({
                "date": self.get_datetime(),
                "action": "Initial Orders Placed",
                "symbol": buy_symbol,
                "buy_price": limit_buy_price,
                "sell_price": limit_sell_price,
                "trail_percent": trail_percent,
                "trail_price": trail_price,
            })

# Main Function
def main():
    # Page Header
    st.markdown('<div class="main-header">📈 Limit Orders & Trailing Stops Strategy</div>', unsafe_allow_html=True)

    # Strategy Description
    st.markdown("""
        **Strategy Overview**  
        This strategy demonstrates how to use limit orders and trailing stops to manage trades:
        - **Limit Orders**: Buy or sell at specific price levels
        - **Trailing Stops**: Automatically adjust stop-loss levels based on price movements
        - **Risk Management**: Protects profits and limits losses
        - **Flexible Configuration**: Adjust parameters for different market conditions
    """)
    st.divider()

    # Configuration Panel
    with st.container():
        st.subheader("⚙️ Strategy Configuration")
        col1, col2 = st.columns(2)
        with col1:
            buy_symbol = st.text_input("Ticker Symbol", value="SPY")
        with col2:
            limit_buy_price = st.number_input("Limit Buy Price", value=403.0)
        
        col3, col4 = st.columns(2)
        with col3:
            limit_sell_price = st.number_input("Limit Sell Price", value=407.0)
        with col4:
            trail_percent = st.number_input("Trailing Stop (%)", value=0.02, format="%.4f")
        
        col5, col6 = st.columns(2)
        with col5:
            trail_price = st.number_input("Trailing Stop ($)", value=7.0)
        with col6:
            live_mode = st.toggle("Live Trading Mode", value=False)

    # Update Strategy Parameters
    LimitAndTrailingStop.parameters = {
        "buy_symbol": buy_symbol,
        "limit_buy_price": limit_buy_price,
        "limit_sell_price": limit_sell_price,
        "trail_percent": trail_percent,
        "trail_price": trail_price,
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
                results = LimitAndTrailingStop.backtest(
                    YahooDataBacktesting,
                    datetime(start_date.year, start_date.month, start_date.day),
                    datetime(end_date.year, end_date.month, end_date.day),
                    benchmark_asset=buy_symbol,
                )
                
                st.subheader("📊 Performance Report")
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric("Strategy Return", f"{results.get('total_return',0):.2f}%" if isinstance(results.get('total_return'), (int, float)) else "N/A")
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
        *Limit orders and trailing stops do not guarantee execution at specified prices. Past performance is not indicative of future results. 
        Trading involves risk of loss. Backtest results are hypothetical and based on historical data.*
        </div>
        """, unsafe_allow_html=True)

# Run the Streamlit App
if __name__ == "__main__":
    # st.set_page_config(page_title="Limit & Trailing Stops Strategy", layout="wide")
    main()