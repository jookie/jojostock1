import streamlit as st
from datetime import datetime, date
from lumibot.entities import Asset, Order
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.example_strategies.drift_rebalancer import DriftRebalancer
from pandas import DataFrame

class DriftRebalancerStrategy(Strategy):
    def initialize(self, cash_at_risk: float = 0.25, drift_threshold: float = 0.05, stock_symbol: str = "SPY", bond_symbol: str = "TLT", stock_allocation: float = 0.60):
        self.set_market("NYSE")
        self.sleeptime = "1D"
        self.cash_at_risk = cash_at_risk
        self.drift_threshold = drift_threshold
        self.stock_symbol = stock_symbol
        self.bond_symbol = bond_symbol
        self.target_weights = {stock_symbol: stock_allocation, bond_symbol: 1 - stock_allocation}
        self.last_trade = None

    def on_trading_iteration(self):
        current_dt = self.get_datetime()
        # Implement rebalance logic here

# Streamlit UI
st.set_page_config(page_title="Drift Rebalancer Portfolio", layout="wide")

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

st.markdown("<div class='stTitle'>📈 Drift Rebalancer Portfolio</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>This strategy automatically rebalances your portfolio when assets drift beyond a set threshold, ensuring your target allocation is maintained.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Asset Selection")
    stock_symbol = st.selectbox("Select Stock ETF", ["SPY", "QQQ", "DIA", "VTI"])
    bond_symbol = st.selectbox("Select Bond ETF", ["TLT", "AGG", "BND", "IEF"])

with col2:
    st.subheader("Allocation Settings")
    stock_allocation = st.slider("Stock Allocation (%)", 10, 90, 60) / 100
    bond_allocation = 1 - stock_allocation  # Ensure total sums to 100%
    st.write(f"Bond Allocation: {bond_allocation * 100:.0f}%")

with col3:
    st.subheader("Rebalancing Parameters")
    drift_threshold = st.slider("Drift Threshold (%)", 1, 10, 5) / 100
    cash_at_risk = st.slider("Cash at Risk (%)", 1, 100, 25) / 100

st.markdown("---")

if st.button("🚀 Run Backtest", use_container_width=True):
    parameters = {
        "market": "NYSE",
        "sleeptime": "1D",
        "drift_threshold": drift_threshold,
        "cash_at_risk": cash_at_risk,
        "target_weights": {stock_symbol: stock_allocation, bond_symbol: bond_allocation},
    }

    results = DriftRebalancer.backtest(
        YahooDataBacktesting,
        datetime.combine(st.date_input("Start Date", date(2023, 1, 2)), datetime.min.time()),
        datetime.combine(st.date_input("End Date", date(2024, 10, 31)), datetime.min.time()),
        benchmark_asset=stock_symbol,
        parameters=parameters,
        show_plot=True,
        show_tearsheet=True,
    )

    st.success("✅ Backtest completed!")
    st.markdown("### 📊 Backtest Results")
    st.write(results)
