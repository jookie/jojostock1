# stock_diversified_leverage.py
from datetime import datetime
import streamlit as st
from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.entities import TradingFee
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader

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
    </style>
    """, unsafe_allow_html=True)

# Page Header
st.markdown('<div class="main-header">📊 Diversified Leveraged Portfolio Strategy</div>', unsafe_allow_html=True)

# Strategy Description
st.markdown("""
    **Strategy Overview**  
    This strategy invests in a diversified portfolio of leveraged ETFs to maximize returns while managing risk:
    - **Diversification**: Spreads investments across multiple asset classes (equities, bonds, commodities)
    - **Leverage**: Uses 2x and 3x leveraged ETFs for amplified returns
    - **Rebalancing**: Automatically rebalances portfolio at regular intervals
    - **Risk Management**: Maintains fixed asset allocation weights
""")
st.divider()

class DiversifiedLeverage(Strategy):
    parameters = {
        "portfolio": [
            {"symbol": "TQQQ", "weight": 0.20},  # 3x Leveraged Nasdaq
            {"symbol": "UPRO", "weight": 0.20},  # 3x Leveraged S&P 500
            {"symbol": "UDOW", "weight": 0.10},  # 3x Leveraged Dow Jones
            {"symbol": "TMF", "weight": 0.25},   # 3x Leveraged Treasury Bonds
            {"symbol": "UGL", "weight": 0.10},   # 3x Leveraged Gold
            {"symbol": "DIG", "weight": 0.15},  # 2x Leveraged Oil and Gas
        ],
        "rebalance_period": 4,  # Rebalance every 4 days
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.counter = None
        self.initialized = False
        self.minutes_before_closing = 1
        st.session_state.portfolio_history = []
        st.session_state.last_rebalance = "No rebalancing yet"

    def on_trading_iteration(self):
        rebalance_period = self.parameters["rebalance_period"]
        if self.counter == rebalance_period or self.counter is None:
            self.counter = 0
            self.rebalance_portfolio()
            st.session_state.last_rebalance = f"🔁 Last rebalanced on {self.get_datetime().strftime('%Y-%m-%d')}"
        
        self.counter += 1

    def rebalance_portfolio(self):
        orders = []
        for asset in self.parameters["portfolio"]:
            symbol = asset["symbol"]
            weight = asset["weight"]
            last_price = self.get_last_price(symbol)
            position = self.get_position(symbol)
            quantity = float(position.quantity) if position else 0
            
            shares_value = self.portfolio_value * weight
            new_quantity = shares_value // last_price
            quantity_difference = new_quantity - quantity
            
            if quantity_difference > 0:
                order = self.create_order(symbol, abs(quantity_difference), "buy")
                orders.append(order)
            elif quantity_difference < 0:
                order = self.create_order(symbol, abs(quantity_difference), "sell")
                orders.append(order)
        
        self.submit_orders(orders)
        st.session_state.portfolio_history.append({
            "date": self.get_datetime(),
            "value": self.portfolio_value,
            "positions": {asset["symbol"]: self.get_position(asset["symbol"]) for asset in self.parameters["portfolio"]}
        })

# UI Configuration Panel
with st.container():
    st.subheader("⚙️ Strategy Configuration")
    col1, col2 = st.columns(2)
    with col1:
        rebalance_period = st.number_input("Rebalance Period (Days)", min_value=1, value=4)
    with col2:
        live_mode = st.toggle("Live Trading Mode", value=False)

# Update Strategy Parameters
DiversifiedLeverage.parameters["rebalance_period"] = rebalance_period

# Backtest Configuration
if not live_mode:
    st.subheader("📅 Backtest Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2010, 6, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2023, 7, 31))

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
    if live_mode:
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = DiversifiedLeverage(broker=broker)
        trader.add_strategy(strategy)
        st.session_state.trader = trader
        st.success("Live trading initialized!")
    else:
        trading_fee = TradingFee(percent_fee=0.005)
        with st.spinner("🔍 Running backtest..."):
            results = DiversifiedLeverage.backtest(
                YahooDataBacktesting,
                datetime(start_date.year, start_date.month, start_date.day),
                datetime(end_date.year, end_date.month, end_date.day),
                benchmark_asset="SPY",
                buy_trading_fees=[trading_fee],
                sell_trading_fees=[trading_fee],
            )
            
            st.subheader("📊 Performance Report")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Strategy Return", f"{results.get('total_return',0):.2f}%" if isinstance(results.get('total_return'), (int, float)) else "N/A")
            with cols[1]:  
                 st.metric("Benchmark Return", f"{results.get('benchmark_return', 0):.2f}%" if isinstance(results.get('benchmark_return'), (int, float)) else "N/A")
            with cols[2]:
                st.metric("Max Drawdown", f"{float(results.get('max_drawdown', {}).get('value', 0)):.2f}%")
            import pandas as pd
            portfolio_data = results.get("portfolio_value")
            if portfolio_data:
                if isinstance(portfolio_data, (list, dict)):  
                    portfolio_df = pd.DataFrame(portfolio_data)

                    if not portfolio_df.empty and len(portfolio_df.columns) > 0:
                        st.line_chart(portfolio_df, use_container_width=True)  # Remove explicit color
                    else:
                        st.warning("⚠️ No portfolio data available for visualization.")
                else:
                    st.warning("⚠️ Unexpected format for portfolio_value.")
            else:
                st.warning("⚠️ No portfolio data found in results.")


# Live Trading Updates
if live_mode and 'trader' in st.session_state:
    st.subheader("📢 Live Portfolio Updates")
    if 'portfolio_history' in st.session_state:
        latest = st.session_state.portfolio_history[-1]
        cols = st.columns(3)
        with cols[0]:
            st.metric("Portfolio Value", f"${latest['value']:,.2f}")
        with cols[1]:
            st.metric("Last Rebalance", latest['date'].strftime("%Y-%m-%d"))
        with cols[2]:
            st.metric("Active Positions", len(latest['positions']))
    
    st.code(st.session_state.last_rebalance)

    if st.button("🛑 Stop Trading"):
        st.session_state.trader.stop_all()
        del st.session_state.trader
        st.success("Trading session stopped!")

# Disclaimer
st.divider()
st.markdown("""
    <div class="disclaimer">
    *Leveraged ETFs are complex instruments that carry significant risk. Past performance is not indicative of future results. 
    Rebalancing does not guarantee profits or protect against losses. Backtest results are hypothetical and based on historical data.*
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    is_live = False
    # ... (original main code remains for script execution)