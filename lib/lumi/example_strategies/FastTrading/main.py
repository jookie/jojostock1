
from datetime import datetime
from lib.MLTradingBot.lumibot.lumibot.backtesting.yahoo_backtesting import YahooDataBacktesting
from lumibot.brokers.alpaca import Alpaca
from lumibot.traders.trader import Trader
from credentials import AlpacaConfig

AlpacaConfig = {
    "API_KEY":  "PKEJH4W0URAU56SHKQW3" ,
    "API_SECRET": "9g6xpk2x2RiBeV5Cy48WdpxCU51chZx91Lj8x6Ow",
    "PAPER": True
}  


logfile = "logs/test.log"
trader = Trader(logfile=logfile)
broker = Alpaca(AlpacaConfig)
# strategy_name = "RedditSentiment" 
# strategy = RedditSentiment(name=strategy_name, budget=budget, broker= broker)
# strategy_name = "DebtTrading" 
# strategy = DebtTrading(name=strategy_name, budget=budget, broker= broker)
strategy_name = "FastTrading" 
strategy = FastTrading(name=strategy_name, budget=budget, broker= broker)
# strategy_name = "My Strategy" 
# strategy = FastTrading(name=strategy_name, budget=budget, broker= broker)
# if type(strategy) != IntrdayMomentum and type(strategy) != FastTrading:
    ###
    # 1. Backtest the strtegy
    ###
backtesting_start = datetime(2012, 1, 1)
backtesting_end   = datetime(2021, 1, 1)

datestring = datetime.now.strftime("%Y-%m-%d %H:%M:%S")
stats_file = f"logs/{strategy_name}_{datestring}.csv"

# Run the actual backtest
print(f"Starting Backtest...")
strategy.backtest(
    YahooDataBacktesting, 
    backtesting_start, 
    backtesting_end,
    stats_file=stats_file
) 
###
# 2. Check benchmark performance
####