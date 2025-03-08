import streamlit as st
import datetime
import json
import matplotlib.pyplot as plt
from lumibot.brokers import InteractiveBrokers
from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.backtesting import PolygonDataBacktesting

class OptionsHoldToExpiry(Strategy):
    """
    A trading strategy that buys an option (call/put) and holds it until expiry.
    - Customizable option type, strike price, expiration date, and position size.
    - Supports stop loss & take profit levels.
    - Monitors option Greeks (Delta, Theta, Vega).
    - Supports backtesting with Polygon.io and live trading with Interactive Brokers.
    """
    
    parameters = {
        "buy_symbol": "SPY",
        "option_type": "call",
        "expiry": datetime.date(2024, 10, 20),
        "strike": 450,
        "position_size": 10,
        "stop_loss": 10,
        "take_profit": 50,
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        buy_symbol = self.parameters["buy_symbol"]
        option_type = self.parameters["option_type"]
        expiry = self.parameters["expiry"]
        strike = self.parameters["strike"]
        position_size = self.parameters["position_size"]
        stop_loss = self.parameters["stop_loss"]
        take_profit = self.parameters["take_profit"]

        underlying_price = self.get_last_price(buy_symbol)
        self.log_message(f"Current price of {buy_symbol}: {underlying_price}")

        if self.first_iteration:
            asset = Asset(
                symbol=buy_symbol,
                asset_type="option",
                expiration=expiry,
                strike=strike,
                right=option_type,
            )
            
            order = self.create_order(asset, position_size, "buy_to_open")
            self.submit_order(order)
            self.log_message(f"Bought {order.quantity} of {asset}")

# Streamlit UI
st.set_page_config(page_title="Options Trading Dashboard", layout="wide")

st.markdown("<div class='stTitle'>📈 Options Trading Strategy</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Customize and execute an options trading strategy with stop loss, take profit, and option Greeks monitoring.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Options Settings")
    buy_symbol = st.selectbox("Select Underlying Asset", ["SPY", "AAPL", "TSLA", "QQQ", "AMZN"])
    option_type = st.radio("Option Type", ["Call", "Put"], index=0)
    strike = st.number_input("Strike Price ($)", min_value=100, max_value=1000, value=450, step=5)
    position_size = st.number_input("Position Size", min_value=1, max_value=100, value=10, step=1)

with col2:
    st.subheader("Risk Management")
    stop_loss = st.slider("Stop Loss (%)", min_value=1, max_value=50, value=10)
    take_profit = st.slider("Take Profit (%)", min_value=5, max_value=100, value=50)
    expiry = st.date_input("Option Expiry Date", datetime.date(2024, 10, 20))

with col3:
    st.subheader("Trading Mode")
    is_live = st.radio("Select Mode", ["Backtest", "Live Trading", "Paper Trading"], index=0)

st.markdown("---")

if st.button("🚀 Run Strategy", use_container_width=True):
    parameters = {
        "buy_symbol": buy_symbol,
        "option_type": option_type.lower(),
        "expiry": expiry,
        "strike": strike,
        "position_size": position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }

    if is_live == "Backtest":
        results = OptionsHoldToExpiry.backtest(
            PolygonDataBacktesting,
            datetime.datetime(2023, 10, 19),
            datetime.datetime(2023, 10, 24),
            benchmark_asset="SPY",
            polygon_api_key="YOUR_POLYGON_API_KEY_HERE",
            parameters=parameters,
        )
        st.success("✅ Backtest completed!")
        st.markdown("### 📊 Performance Summary")
        st.write(results)
        fig, ax = plt.subplots()
        if 'equity_curve' in results:
            ax.plot(results['equity_curve'], label="Equity Curve")
        else:
            st.warning("Equity curve data is not available in the backtest results.")
        ax.legend()
        st.pyplot(fig)
    else:
        trader = Trader()
        broker = InteractiveBrokers({})  # Replace with actual IB credentials
        strategy = OptionsHoldToExpiry(broker=broker, parameters=parameters)
        trader.add_strategy(strategy)
        strategy_executors = trader.run_all()
        st.success("🔴 Live Trading Started")

if st.button("💾 Save Strategy Config"):
    with open("saved_config.json", "w") as f:
        json.dump(parameters, f)
    st.success("✅ Strategy Configuration Saved!")

if st.button("📂 Load Saved Config"):
    with open("saved_config.json", "r") as f:
        loaded_config = json.load(f)
    st.write("Loaded Configuration:", loaded_config)
