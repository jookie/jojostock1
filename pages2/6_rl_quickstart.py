# pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

# https://finrl.readthedocs.io/en/latest/reference/reference.html
# https://finrl.readthedocs.io/en/latest/start/quick_start.html
# https://finrl.reaqq``````````````````````````qqqdthedocs.io/en/latest/start/installation.html

import os
import sys
sys.path.append("lib")
from typing import List
from argparse import ArgumentParser
from lib.rl.config_tickers import DOW_30_TICKER
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
API_BASE_URL = 'https://paper-api.alpaca.markets'

from lib.rl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)

# construct environment
from lib.rl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from lib.rl import config

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


# "./" will be added in front of each directory
def check_and_make_directories(directories: List[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)



def main():
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    if options.mode == "train":
        from lib.rl.train import train

        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            # data_source="yahoofinance",
            data_source="alpaca",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            erl_params=ERL_PARAMS,
            break_step=1e5,
            kwargs=kwargs,
        )
    elif options.mode == "test":
        from lib.rl import test
        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

        account_value_erl = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            # data_source="yahoofinance",
            data_source="alpaca",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            net_dimension=512,
            kwargs=kwargs,
        )
    elif options.mode == "trade":
        from lib.rl import trade
        env = StockTradingEnv
        kwargs = {}
        trade(
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=DOW_30_TICKER,
            # data_source="yahoofinance",
            data_source="alpaca",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            API_KEY=ALPACA_API_KEY,
            API_SECRET=ALPACA_API_SECRET,
            API_BASE_URL=ALPACA_API_BASE_URL,
            trade_mode='backtesting',
            if_vix=True,
            kwargs=kwargs,
        )
    else:
        raise ValueError("Wrong mode.")


## Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    main()