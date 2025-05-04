from datetime import datetime
from lumibot.backtesting.alpaca_backtesting import AlpacaBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.entities import Asset

from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
ALPACA_CONFIG  = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True  # Set to True for paper trading
}

class BuyAndHold(Strategy):
    parameters = {
        "symbol": "SPY",  # Asset to trade
    }

    def initialize(self):
        self.sleeptime = "1D"  # Run strategy once per day

    def on_trading_iteration(self):
        if self.first_iteration:
            symbol = self.parameters["symbol"]
            price = self.get_last_price(symbol)
            if price is None:
                print(f"Unable to get price for {symbol}. Skipping buy order.")
                return
            # Calculate quantity based on portfolio value
            quantity = int(self.portfolio_value / price)
            if quantity > 0:
                order = self.create_order(symbol, quantity, "buy")
                self.submit_order(order)
                print(f"Bought {quantity} shares of {symbol} at {price}")

if __name__ == "__main__":
    # Define backtesting period
    backtesting_start = datetime(2023, 1, 1)
    backtesting_end = datetime(2025, 5, 1)

    # Initialize Alpaca broker for backtesting
    broker = Alpaca(ALPACA_CONFIG)

    # Define the asset for benchmarking
    benchmark_asset = Asset(symbol="SPY", asset_type="stock")

    # Run the backtest
    result = BuyAndHold.backtest(
        AlpacaBacktesting,
        backtesting_start,
        backtesting_end,
        broker=broker,
        benchmark_asset=benchmark_asset,
        budget=10000,  # Initial portfolio value
        parameters={"symbol": "SPY"},
        stats_file="logs/buy_and_hold_stats.csv",
        plot_file_html="logs/buy_and_hold_plot.html",
        trades_file="logs/buy_and_hold_trades.csv",
        save_tearsheet=True,
        tearsheet_file="logs/buy_and_hold_tearsheet.html"
    )

    # Print the backtest results
    print(result)