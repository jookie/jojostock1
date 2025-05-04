# stock_momentum.py
import sys
# sys.path.append('/Users/dovpeles/dov/jojostock1/lib/MLTradingBot')
from lumibot.strategies.strategy import Strategy
from datetime import datetime
import streamlit as st
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
st.markdown('<div class="main-header">📈 Momentum Trading Strategy</div>', unsafe_allow_html=True)

# Strategy Description
st.markdown("""
    **Strategy Overview**  
    This strategy identifies the best-performing asset from a predefined list over a specified period and invests in it:
    - **Momentum-Based**: Buys the asset with the highest recent returns
    - **Daily Rebalancing**: Adjusts holdings based on daily performance
    - **Diversified Assets**: Tracks multiple symbols (e.g., SPY, VEU, AGG)
    - **Risk Management**: Automatically reallocates to the best-performing asset
""")
st.divider()

# Momentum Strategy Class
class Momentum(Strategy):
    def initialize(self, symbols=None):
        self.period = 2
        self.counter = 0
        self.sleeptime = 0
        self.symbols = symbols if symbols else ["SPY", "VEU", "AGG"]
        self.asset = ""
        self.quantity = 0
        st.session_state.trade_history = []
        st.session_state.last_trade = "No trades yet"

    def on_trading_iteration(self):
        if self.counter == self.period or self.counter == 0:
            self.counter = 0
            momentums = self.get_assets_momentums()

            # Find the best-performing asset
            momentums.sort(key=lambda x: x.get("return"))
            best_asset_data = momentums[-1]
            best_asset = best_asset_data["symbol"]
            best_asset_return = best_asset_data["return"]

            # Check if the best asset has changed
            if self.asset:
                current_asset_data = [m for m in momentums if m["symbol"] == self.asset][0]
                current_asset_return = current_asset_data["return"]
                if current_asset_return >= best_asset_return:
                    best_asset = self.asset
                    best_asset_data = current_asset_data

            # Log the best asset
            self.log_message(f"{best_asset} is the best-performing asset.")

            # Rebalance if necessary
            if best_asset != self.asset:
                if self.asset:
                    self.log_message(f"Swapping {self.asset} for {best_asset}.")
                    order = self.create_order(self.asset, self.quantity, "sell")
                    self.submit_order(order)
                    st.session_state.last_trade = f"🔄 Swapped {self.asset} for {best_asset}"

                self.asset = best_asset
                best_asset_price = best_asset_data["price"]
                self.quantity = int(self.portfolio_value // best_asset_price)
                order = self.create_order(self.asset, self.quantity, "buy")
                self.submit_order(order)
                st.session_state.last_trade += f"\n🛒 Bought {self.quantity} shares of {best_asset} at ${best_asset_price:.2f}"
            else:
                self.log_message(f"Keeping {self.quantity} shares of {self.asset}.")
                st.session_state.last_trade = f"📊 Holding {self.quantity} shares of {self.asset}"

            st.session_state.trade_history.append({
                "date": self.get_datetime(),
                "action": "Rebalanced",
                "symbol": best_asset,
                "quantity": self.quantity,
                "price": best_asset_price,
                "return": best_asset_return,
            })

        self.counter += 1
        self.await_market_to_close()

    def on_abrupt_closing(self):
        self.sell_all()

    def get_assets_momentums(self):
        momentums = []
        data = self.get_bars(self.symbols, self.period + 2, timestep="day")
        for asset, bars_set in data.items():
            symbol = asset.symbol
            symbol_momentum = bars_set.get_momentum(num_periods=self.period)
            momentums.append({
                "symbol": symbol,
                "price": bars_set.get_last_price(),
                "return": symbol_momentum,
            })
        return momentums

# Configuration Panel
with st.container():
    st.subheader("⚙️ Strategy Configuration")
    col1, col2 = st.columns(2)
    with col1:
        symbols = st.text_input("Symbols (comma-separated)", value="SPY,VEU,AGG")
        symbols = [s.strip() for s in symbols.split(",")]
    with col2:
        period = st.number_input("Momentum Period (Days)", min_value=1, value=2)
    
    live_mode = st.toggle("Live Trading Mode", value=False)

# Backtest Configuration
if not live_mode:
    st.subheader("📅 Backtest Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2023, 8, 1))

# Execution Control
if st.button("🚀 Start Analysis" if live_mode else "📊 Run Backtest"):
    if live_mode:
        st.error("Live trading is not implemented in this example.")
    else:
        with st.spinner("🔍 Running backtest..."):
            results = Momentum.backtest(
                YahooDataBacktesting,
                datetime(start_date.year, start_date.month, start_date.day),
                datetime(end_date.year, end_date.month, end_date.day),
                benchmark_asset="SPY",
                parameters={"symbols": symbols, "period": period},
            )
            
            st.subheader("📊 Performance Report")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Strategy Return", f"{results.get('total_return', 0):.2f}%")
            with cols[1]:
                benchmark_return = results.get("benchmark_return", "N/A")
                if benchmark_return != "N/A":
                    st.metric("Benchmark Return", f"{benchmark_return:.2f}%")
                else:
                    st.metric("Benchmark Return", "N/A")
            with cols[2]:
                st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2f}%")
            
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
    *Momentum trading involves substantial risk and may not be suitable for all investors. 
    Past performance is not indicative of future results. Backtest results are hypothetical and based on historical data.*
    </div>
    """, unsafe_allow_html=True)