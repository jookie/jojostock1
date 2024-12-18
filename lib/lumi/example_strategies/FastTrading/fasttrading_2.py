from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from lib.MLTradingBot.t1 import strategy
from lib.MLTradingBot.lumibot.lumibot.example_strategies.fasttrading_2 import strategy.name

class FastTrading(Strategy):
    IS_BACKTESTING = False
    
    # ===== Overloading Lifecycle Methods =====
    def initialize(self, momentum_length=2, max_assets=4):
        # Set symbols we want to monitor
        self.symbols = ['TSLA', 'SPY', 'GLD', 'TLT', 'MSFT', 'MCHI', 'SPXL', 'SPXS']
        self.momentum_length = momentum_length  # in minutes
        self.sleeptime = 1  # Optional: For slowing down execution
        self.frequency = "minute"  # For minute-level trading
        self.max_assets = min(max_assets, len(self.symbols))  # Limit max assets to trade
        self.position_size = 1  # Capital per trade (1 unit for simplicity)

        # Initialize variables
        self.assets_quantity = {symbol: 0 for symbol in self.symbols}
        self.quantity = 0  # Quantity to buy per trade

        # Schedule periodic trades
        self.scheduled_task(self.trade, time_interval=timedelta(minutes=self.momentum_length))

    def trade(self):
        # Step 1: Fetch historical data for all symbols
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=self.momentum_length)
        historical_data = {}

        for symbol in self.symbols:
            try:
                # Fetch historical data using the broker's API
                historical_data[symbol] = self.get_historical_data(
                    symbol,
                    start=start_time,
                    end=end_time,
                    granularity="minute"
                )
            except Exception as e:
                self.log_message(f"Error fetching data for {symbol}: {e}")
                continue

        # Step 2: Calculate momentum (percentage change) for each symbol
        momentum_scores = {}
        for symbol, data in historical_data.items():
            if data is not None and len(data) >= 2:  # Ensure we have enough data
                first_price = data['close'].iloc[0]
                last_price = data['close'].iloc[-1]
                if first_price > 0:  # Avoid division by zero
                    momentum_scores[symbol] = ((last_price - first_price) / first_price) * 100

        # Step 3: Identify top performers (up to max_assets)
        sorted_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_performers = sorted_momentum[:self.max_assets]

        # Log the momentum scores
        self.log_message(f"Momentum scores: {momentum_scores}")
        self.log_message(f"Top performers: {top_performers}")

        # Step 4: Execute trades (buy top-performing assets)
        for symbol, momentum in top_performers:
            try:
                # Calculate how much to buy (e.g., fixed position size)
                self.market_order(symbol, quantity=self.position_size)
                self.log_message(f"Bought {self.position_size} of {symbol} (momentum: {momentum:.2f}%)")
            except Exception as e:
                self.log_message(f"Error executing trade for {symbol}: {e}")
    
    def on_trading_iteration(self):
        # Optional: Implement if there's logic to execute on every iteration
        self.log_message("Trading iteration executed.")
        buying_budget = self.unspent_money
