from __future__ import annotations
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)
# print(f"DEBUG: Added path: {project_root}")
import warnings
warnings.filterwarnings("ignore")
#+++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++
from lib.rl.config_tickers import DOW_30_TICKER
ticker_list = DOW_30_TICKER
print(ticker_list)
from lib.rl.config import DATA_API_KEY ,DATA_API_SECRET, DATA_API_BASE_URL
#+++++++++++++++++++++++++++++++++++++++++
import argparse

from lib.rl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from lib.rl.config import INDICATORS
# Import Dow Jones 30 Symbols

#    lib.rl/meta/paper_trading/common.py
from lib.rl.meta.paper_trading.common import train, test, alpaca_history, DIA_history
from lib.rl.meta.paper_trading.alpaca import PaperTradingAlpaca

env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

# Set up sliding window of 6 days training and 2 days testing
import datetime
from pandas.tseries.offsets import BDay  # BDay is business day, not birthday...

today = datetime.datetime.today()

TEST_END_DATE = (today - BDay(1)).to_pydatetime().date()
TEST_START_DATE = (TEST_END_DATE - BDay(1)).to_pydatetime().date()
TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
TRAIN_START_DATE = (TRAIN_END_DATE - BDay(5)).to_pydatetime().date()
TRAINFULL_START_DATE = TRAIN_START_DATE
TRAINFULL_END_DATE = TEST_END_DATE

TRAIN_START_DATE = str(TRAIN_START_DATE)
TRAIN_END_DATE = str(TRAIN_END_DATE)
TEST_START_DATE = str(TEST_START_DATE)
TEST_END_DATE = str(TEST_END_DATE)
TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("TRAIN_END_DATE: ", TRAIN_END_DATE)
print("TEST_START_DATE: ", TEST_START_DATE)
print("TEST_END_DATE: ", TEST_END_DATE)
print("TRAINFULL_START_DATE: ", TRAINFULL_START_DATE)
print("TRAINFULL_END_DATE: ", TRAINFULL_END_DATE)
