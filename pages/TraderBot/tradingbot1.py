# tradingbot.py
from lumibot.backtesting import BacktestingBroker
from lumibot.data_sources import AlpacaData
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca

from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
import quantstats as qs
# from lumibot.backtesting.alpaca_backtesting import WorkingAlpacaBacktesting
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
from lumibot.brokers import Alpaca
from alpaca_trade_api import REST 

BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
    "API_KEY":ALPACA_API_KEY, 
    "API_SECRET": ALPACA_API_SECRET, 
    "PAPER": True
}


BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
    "API_KEY":ALPACA_API_KEY, 
    "API_SECRET": ALPACA_API_SECRET, 
    "PAPER": True
}

from lumibot.strategies import Strategy
from datetime import datetime
from lumibot.backtesting.alpaca_backtesting import WorkingAlpacaBacktesting

class SimpleAlpacaStrategy(Strategy):
    def initialize(self):
        self.symbol = "SPY"
        self.sleeptime = "1D"
        
    def on_trading_iteration(self):
        # Get historical data
        bars = self.get_historical_prices(self.symbol, 20, "day")
        current_price = bars.df["close"][-1]
        
        # Simple buy and hold strategy
        if not self.get_position(self.symbol):
            quantity = self.portfolio_value // current_price
            if quantity > 0:
                order = self.create_order(self.symbol, quantity, "buy")
                self.submit_order(order)

# Backtest configuration
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)



from lumibot.backtesting.alpaca_backtesting import WorkingAlpacaBacktesting
from lumibot.backtesting import BacktestingBroker


# Create data source
data_source = WorkingAlpacaBacktesting(
    datetime_start=start_date,
    datetime_end=end_date,   # Add these if you need real Alpaca data:
    API_KEY=ALPACA_API_KEY,
    API_SECRET=ALPACA_API_SECRET
)

# Run backtest
strategy = SimpleAlpacaStrategy(
    broker=data_source,
    budget=100000
)

results = strategy.backtest()
print(f"Backtest results: {results}")