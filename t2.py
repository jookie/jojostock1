
from __future__ import annotations
import warnings ; warnings.filterwarnings("ignore")
from lumibot.strategies.strategy import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from datetime import datetime

from lumibot.data_sources import AlpacaData


from lumibot.backtesting.alpaca_backtesting import AlpacaBacktesting

class BuyAndHold(Strategy):
    def initialize(self):
        # Set the symbol to trade (e.g., SPY for S&P 500 ETF)
        self.symbol = "SPY"
        # Set how often to run the trading logic (daily)
        self.sleeptime = "1D"
        
    def on_trading_iteration(self):
        # Only buy on the first iteration
        if self.first_iteration:
            # Get the current price of the asset
            current_price = self.get_last_price(self.symbol)
            # Calculate quantity based on portfolio value
            quantity = int(self.portfolio_value // current_price)
            # Place buy order
            self.buy_order = self.create_order(
                self.symbol,
                quantity,
                "buy",
                type="market"
            )
            self.submit_order(self.buy_order)
            self.log_message(f"Bought {quantity} shares of {self.symbol} at ${current_price:.2f}")

if __name__ == "__main__":
    # Define backtesting period
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Alpaca configuration (replace with your credentials)
    from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
    alpaca_config = {
        "API_KEY": ALPACA_API_KEY,
        "API_SECRET": ALPACA_API_SECRET,
        "PAPER": True  # Set to True for paper trading
    }
    
    # Initialize trader and strategy
    trader = Trader()
    broker = Alpaca(alpaca_config)
    strategy = BuyAndHold(
        broker=broker,
        parameters={}
    )
   
    # Run backtest
    backtesting = AlpacaBacktesting(
        start_date=start_date,
        end_date=end_date,
        data_source="alpaca",
        api_key=alpaca_config["API_KEY"],
        api_secret=alpaca_config["API_SECRET"],
        paper=alpaca_config["PAPER"]
    )
    
    trader.backtest(
        strategy=strategy,
        backtesting_engine=backtesting,
        show_plot=True,
        save_results=True
    )