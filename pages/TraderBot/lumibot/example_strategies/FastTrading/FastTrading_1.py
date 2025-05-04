
"""""
https://github.com/mjmacarty/alphavantage/blob/main/3-momentum_algorithmic.ipynb
"Momentum" Trading Strategy
Sometimes referred to as trend following where you 
Buys the best performing asset from self.symbols over self.momentum_length number of minutes. 
For Example, if TSLA increased 0,03% in the past two minutes, 
but SPY , GLD, TLT and MSFT only increase 0.01% in the pasdt two minutes, 
then we will but TSLA.
To accomplish the described task, I'll implement 
a trading strategy that buys the best-performing asset 
based on momentum over a specified time period. 
I will use `lumibot`, and a trading framework called - Alpaca,
"""""

# To convert a Jupyter Notebook file (.ipynb) to a Python script (.py), 
# you can use Python's built-in nbconvert utility, 
# which is part of the nbformat library. 
# Here’s a step-by-step explanation:
# jupyter nbconvert --to script your_notebook.ipynb


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import logging
import datetime
from lumibot.strategies.strategy import Strategy
import pandas_datareader as pdr
from lib.rl.plot import plot_result ; 
logger = logging.getLogger(__name__)
class Developing_Momentum_Trading_Strategy:
    # https://github.com/mjmacarty/alphavantage/blob/main/3-momentum_algorithmic.ipynb
    # Many services for this, some paid some free
    # Yahoo Finance API
    # Typically trading "systems" involve a number of securities
    # For this demonstration we are just going to look at GLD --> the gold ETF
    import numpy as np
    import pandas as pd
    import pandas_datareader as pdr
    import matplotlib.pyplot as plt   
    
    def  Download_Data():
        # https://github.com/mjmacarty/alphavantage/blob/main/3-momentum_algorithmic.ipynb
        # Many services for this, some paid some free
        # Yahoo Finance API
        # Typically trading "systems" involve a number of securities
        # For this demonstration we are just going to look at GLD --> the gold ETF
    
        gld = pdr.get_data_yahoo('GLD')
        day = np.arange(1, len(gld) + 1)
        gld['day'] = day
        gld.drop(columns=['Adj Close', 'Volume'], inplace = True)
        gld = gld[['day', 'Open', 'High', 'Low', 'Close']]
        gld.head()
        #     day	Open	High	Low	Close
        # Date					
        # 2016-04-11	1	119.470001	120.300003	119.419998	120.029999
        # 2016-04-12	2	120.230003	120.349998	119.550003	120.050003
        # 2016-04-13	3	119.370003	119.430000	118.559998	118.769997
        # 2016-04-14	4	118.089996	118.190002	116.949997	117.110001
        # 2016-04-15	5	117.330002	118.120003	117.190002	117.919998
        gld.info()
        # <class 'pandas.core.frame.DataFrame'>
        # DatetimeIndex: 1258 entries, 2016-04-11 to 2021-04-08
        # Data columns (total 5 columns):
        # #   Column  Non-Null Count  Dtype  
        # ---  ------  --------------  -----  
        # 0   day     1258 non-null   int64  
        # 1   Open    1258 non-null   float64
        # 2   High    1258 non-null   float64
        # 3   Low     1258 non-null   float64
        # 4   Close   1258 non-null   float64
        # dtypes: float64(4), int64(1)
        # memory usage: 59.0 KB
        
    def  AddDataTransformData(gld):
        # https://github.com/mjmacarty/alphavantage/blob/main/3-momentum_algorithmic.ipynb
        calculate signal based on some price or statistical action
        # we are going to try a moving average crossover to generate signals
        # for this strategy we will always by "in" a trade, either long or short
        # we are modeling; this means real life variation should be expected
        gld['9-day'] = gld['Close'].rolling(9).mean()
        gld['21-day'] = gld['Close'].rolling(21).mean()
        gld[19:25]
        # day	Open	High	Low	Close	9-day	21-day	signal	return	system_return	entry
        # Date											
        # 2016-06-07	41	118.500000	119.019997	118.410004	118.820000	117.052222	NaN	-1	-0.000841	0.000841	0.0
        # 2016-06-08	42	120.300003	120.779999	120.230003	120.580002	117.452222	119.066192	-1	0.014704	-0.014704	0.0
        # 2016-06-09	43	120.610001	121.480003	120.550003	121.250000	117.971111	119.079049	-1	0.005541	-0.005541	0.0
        # 2016-06-10	44	121.550003	122.099998	121.180000	121.739998	118.651110	119.060953	-1	0.004033	-0.004033	0.0
        # 2016-06-13	45	122.800003	122.830002	122.029999	122.639999	119.382222	119.131429	-1	0.007366	-0.007366	0.0
        # 2016-06-14	46	123.000000	123.059998	122.300003	122.769997	120.141110	119.181905	1	0.001059	0.001059	2.0
    
    def Add_signal_column(gld): 
        gld['signal'] = np.where(gld['9-day'] > gld['21-day'], 1, 0)
        gld['signal'] = np.where(gld['9-day'] < gld['21-day'], -1, gld['signal'])
        gld.dropna(inplace=True)
        gld.head()   
        #              day	Open	    High	    Low	        Close	    9-day	    21-day	 signal return	 system_return	entry
        # Date											
        # 2016-06-08	42	120.300003	120.779999	120.230003	120.580002	117.452222	119.066192	-1	0.014704	-0.014704	0.0
        # 2016-06-09	43	120.610001	121.480003	120.550003	121.250000	117.971111	119.079049	-1	0.005541	-0.005541	0.0
        # 2016-06-10	44	121.550003	122.099998	121.180000	121.739998	118.651110	119.060953	-1	0.004033	-0.004033	0.0
        # 2016-06-13	45	122.800003	122.830002	122.029999	122.639999	119.382222	119.131429	1	0.007366	-0.007366	0.0
        # 2016-06-14	46	123.000000	123.059998	122.300003	122.769997	120.141110	119.181905	1	0.001059	0.001059	2.0

    def Calculate_Instantaneous_returns_system_returns(gld):
        gld['return'] = np.log(gld['Close']).diff()
        gld['system_return'] = gld['signal'] * gld['return']
        gld['entry'] = gld.signal.diff()
        gld.head()
    
    def Plot_trades_on_time_series(gld):
        plt.rcParams['figure.figsize'] = 12, 6
        plt.grid(True, alpha = .3)
        plt.plot(gld.iloc[-252:]['Close'], label = 'GLD')
        plt.plot(gld.iloc[-252:]['9-day'], label = '9-day')
        plt.plot(gld.iloc[-252:]['21-day'], label = '21-day')
        plt.plot(gld[-252:].loc[gld.entry == 2].index, gld[-252:]['9-day'][gld.entry == 2], '^',
                color = 'g', markersize = 12)
        plt.plot(gld[-252:].loc[gld.entry == -2].index, gld[-252:]['21-day'][gld.entry == -2], 'v',
                color = 'r', markersize = 12)
        plt.legend(loc=2);
        # =========plot_result======
        plt.plot(np.exp(gld['return']).cumprod(), label='Buy/Hold')
        plt.plot(np.exp(gld['system_return']).cumprod(), label='System')
        plt.legend(loc=2)
        plt.grid(True, alpha=.3)    
        # =========plot_result======
        np.exp(gld['return']).cumprod()[-1] -1
        # 0.3653996523304317
        np.exp(gld['system_return']).cumprod()[-1] -1
        # 0.6523109676509895

class FastTrading(Strategy):
    
    # =========over loading life cycle methods
    
    def initialize(self, momentum_length = 2, max_assets = 4):
       
        self.momentum_length =  momentum_length # in minutes
        self.sleeptime = 1
        
         # set symbols tht we want to be monitoring
        self.symbols = ['TSLA', 'SPY', 'GLD', 'TLT', 'MSFT', 'MCHI', 'SPXL', 'SPXS']
        
        # Initialise our variables  
        self.assets_quantity = {symbol:0 for symbol in self.symbols}
        self.max_assets = min(max_assets, len(self.symbols))
        self.quantity = 0
        
    def on_trading_iteration(self):   
        # Setting the buyingd budget
        buying_budget =  self.unspent_money
        
        # Get the momentums of all the assets we are tracking
        momentums = self.get_assets_momentums()
        for item in momentums:
            symbol = item.get(symbol)
            if self.assets_quantity[symbol] > 0:
                item["held"] = True
            else:
                item['held'] = False
                
        # Get the assets with the highest return in our momentum momentum_length
        # (aka the highest momentum)
        # In case of parity , giving priority to the current asset 
        momentums.sort(key=lambda x: (x.get("return"), x.get("held")))
        prices = {item.get("symbol"): item.get("price") for item in momentums}   
        best_assets = momentums[-self.max_assets :]    
        best_assets_symbols = [item.get("symbol") for item in best_assets]    
        
        # Deciding which assets to keep, sell and buy  
        assets_to_keep = []
        assets_to_sell = []
        assets_to_buy  = []
        for symbol, quantity in self.assets_quantity.items():
            if quantity > 0 and symbol in best_assets_symbols:
                # The asset is still a top asset and should be kept
                assets_to_keep.append(symbol)
            elif quantity <= 0 and symbol in best_assets_symbols:
                # Need to buy this new asset
                assets_to_buy.append(symbol)
            elif quantity > 0 and symbol not in best_assets_symbols:
                # The asset is no longer s top asset and should br sold
                assets_to_sell.append(symbol)    
    # ========Helper methods============    
    def get_assets_momentum(self):
        
        """
        Gets the momentums (the percentage return) for all trhe assets we are tracking,
        over the time period set in self.momentum_length
        """  
        momentums = []
     
            
        return momentums
   
    def trace_stats(self, context, snapshot_before):
        """
        Add additional stats to the CSV logfile
        """
        # Get the values of all our variables from the last iteration
        row = {
            "old_best_asset": snapshot_before.get("asset"),
            "old_asset_quantity": snapshot_before.get("quantity"),
            "old_cash": snapshot_before.get("cash"),
            "new_best_asset": self.asset,
            "new_asset_quantity": self.quantity,
        }

        # Get the momentums of all the assets from the context of on_trading_iteration
        # (notice that on_trading_iteration has a variable called momentums, this is what
        # we are reading here)
        momentums = context.get("momentums")
        if len(momentums) != 0:
            for item in momentums:
                symbol = item.get("symbol")
                for key in item:
                    if key != "symbol":
                        row[f"{symbol}_{key}"] = item[key]

        # Add all of our values to the row in the CSV file. These automatically get
        # added to portfolio_value, cash and return
        return row
         
    def before_market_closes(self):    
        # Make sure that we sell everything before the market closes
        self.sell_all()
        self.quantity = 0
        self.assets_quantity = {symbol:0 for symbol in self.symbols}
    
    def on_abrupt_closing(self):    
        # Make sure that we sell everything before the market closes
        self.sell_all()
        self.quantity = 0
        self.assets_quantity = {symbol:0 for symbol in self.symbols}   
        
    def test(self):
        for symbol in self.symbols:
            if quantity > 0 and symbol not in best_assets_symbols:
                # Need to buy this asset
                assets_to_buy.append(symbol)
            if quantity > 0 and symbol not in best_assets_symbols:
                # The asset is no longer s top asset and should br sold
                assets_to_sell.append(symbol)
                
        # Printing deceision         
        self.log_message("Selling %r" % asssets_to_sell)
        self.log_message("Buying %r" % asssets_to_buy)   
        
        # selling assets
        selling_orders = []
        for symbol in assert_to_sell:
            self.log_message("Selling %s. symbol")
            quantity = self.assets_quantity[symbol]
            order = self.create_order(symbol, quantity, "sell")
            selling_orders.append(order)
        self.submit_order(selling_orders)  
        self.wait_for_order_execution(selling_orders)  
        
        # Checking if all orders went successfuly through
        assets_sold = 0   
        for order in selling_orders:
            if order.status == "fill":
                self.assets_quantity[order.symbol] = 0
                assets_sold += 1   
                buying_budget += order.quantity *  prices.get(order.symbol)
            
        # Buying new assets
        if self.first_iteration:
            number_of_assets_to_buy = self.max_assets
        else:
            number_of_assets_to_buy = assets_sold      
            
        for i in range(number_of_assets_to_buy):
            symbol = assets_to_buy[i]
            price  = prices.get(symbol) 
            quantity = (number_of_assets_to_buy) // price 
            order = self.create_order(symbol, quantity, "buy")  
            self.log_message("Buying %d shares of %s." % (quantity, symbol))
            self.submit_order(order)
            self.assets_quantity[symbol] = quantity

if __name__ == "__main__":
    IS_BACKTESTING = True

    if IS_BACKTESTING:
        from lumibot.backtesting import YahooDataBacktesting

        # Backtest this strategy
        backtesting_start = datetime.datetime(2023, 1, 1)
        backtesting_end = datetime.datetime(2024, 9, 1)

        results = FastTrading.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            benchmark_asset="SPY",
            # show_progress_bar=False,
            # quiet_logs=False,
        )

        # Print the results
        print(results)
    else:
        from lumibot.credentials import ALPACA_CONFIG
        from lumibot.brokers import Alpaca
        from lumibot.traders import Trader

        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = FastTrading(broker=broker)
        trader.add_strategy(strategy)
        strategy_executors = trader.run_all()
