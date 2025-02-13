import streamlit as st
from datetime import datetime, date
from lumibot.entities import Asset, Order
from lumibot.backtesting import CcxtBacktesting
from lumibot.strategies.strategy import Strategy
from pandas import DataFrame

class CcxtBacktestingExampleStrategy(Strategy):
    def initialize(self, asset: tuple[Asset, Asset] = None,
                   cash_at_risk: float = .25, window: int = 21):
        if asset is None:
            raise ValueError("You must provide a valid asset pair")
        self.set_market("24/7")
        self.sleeptime = "1D"
        self.asset = asset
        self.base, self.quote = asset
        self.window = window
        self.symbol = f"{self.base.symbol}/{self.quote.symbol}"
        self.last_trade = None
        self.order_quantity = 0.0
        self.cash_at_risk = cash_at_risk

    def on_trading_iteration(self):
        current_dt = self.get_datetime()
        cash, last_price, quantity = self._position_sizing()
        history_df = self._get_historical_prices()
        bbands = self._get_bbands(history_df)
        prev_bbp = bbands[bbands.index < current_dt].tail(1).bbp.values[0]

        if prev_bbp < -0.13 and cash > 0 and self.last_trade != Order.OrderSide.BUY and quantity > 0.0:
            order = self.create_order(self.base, quantity, side=Order.OrderSide.BUY,
                                      type=Order.OrderType.MARKET, quote=self.quote)
            self.submit_order(order)
            self.last_trade = Order.OrderSide.BUY
            self.order_quantity = quantity
        elif prev_bbp > 1.2 and self.last_trade != Order.OrderSide.SELL and self.order_quantity > 0.0:
            order = self.create_order(self.base, self.order_quantity, side=Order.OrderSide.SELL,
                                      type=Order.OrderType.MARKET, quote=self.quote)
            self.submit_order(order)
            self.last_trade = Order.OrderSide.SELL
            self.order_quantity = 0.0

# Streamlit UI
st.set_page_config(layout="wide")

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
        .stMarkdown {
            text-align: center;
            font-size: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='stTitle'>📈 Cryptocurrency Backtesting Tool</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Bollinger Bands are a tool used to analyze price trends. They consist of three lines: an average price in the middle, an upper band, and a lower band. When Bitcoin's price moves close to the lower band, it may be undervalued (a buy signal), and when it reaches the upper band, it may be overvalued (a sell signal). This strategy helps automate trading based on these principles.</p>", unsafe_allow_html=True)
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    base_symbol = st.selectbox("Select Base Currency", ["BTC", "ETH", "ADA", "SOL"])
    quote_symbol = st.selectbox("Select Quote Currency", ["USDT", "USD", "BTC"])
with col2:
    exchange_id = st.selectbox("Select Exchange", ["kraken", "binance", "bybit", "okx", "bitmex", "kucoin"])
    cash_at_risk = st.slider("Cash at Risk (%)", 1, 100, 25) / 100
with col3:
    col3_1, col3_2 = st.columns(2)
    with col3_1:
        start_date = datetime.combine(st.date_input("Start Date", date(2023, 2, 11)), datetime.min.time())
    with col3_2:
        end_date = datetime.combine(st.date_input("End Date", date(2024, 2, 12)), datetime.min.time())
    window = st.slider("Bollinger Bands Window", 5, 50, 21)

st.markdown("---")

if st.button("🚀 Run Backtest", use_container_width=True):
    asset = (Asset(symbol=base_symbol, asset_type="crypto"),
             Asset(symbol=quote_symbol, asset_type="crypto"))
    kwargs = {"exchange_id": exchange_id}
    CcxtBacktesting.MIN_TIMESTEP = "day"
    
    results, strat_obj = CcxtBacktestingExampleStrategy.run_backtest(
        CcxtBacktesting,
        start_date,
        end_date,
        benchmark_asset=f"{base_symbol}/{quote_symbol}",
        quote_asset=Asset(symbol=quote_symbol, asset_type="crypto"),
        parameters={"asset": asset, "cash_at_risk": cash_at_risk, "window": window},
        **kwargs,
    )
    st.success("✅ Backtest completed!")
    st.markdown("### 📊 Backtest Results")
    st.write(results)
