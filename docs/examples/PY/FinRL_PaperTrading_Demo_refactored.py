
# install finrl library
# %pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git
# Alpaca keys
from __future__ import annotations

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("data_key", help="data source api key")
# parser.add_argument("data_secret", help="data source api secret")
# parser.add_argument("data_url", help="data source api base url")
# parser.add_argument("trading_key", help="trading api key")
# parser.add_argument("trading_secret", help="trading api secret")
# parser.add_argument("trading_url", help="trading api base url")
# args = parser.parse_args()
DATA_API_KEY = "PKVD6WOSPEMKS0UI6A3K" #args.data_key
DATA_API_SECRET = "BxT64PIQtDBb*tnW"  #args.data_secret
DATA_API_BASE_URL = 'https://paper-api.alpaca.markets' #args.data_url
TRADING_API_KEY = "PKVD6WOSPEMKS0UI6A3K" #args.trading_key
TRADING_API_SECRET = "BxT64PIQtDBb*tnW"  #args.trading_secret
TRADING_API_BASE_URL = 'https://paper-api.alpaca.markets' #args.trading_url

print("DATA_API_KEY: ", DATA_API_KEY)
print("DATA_API_SECRET: ", DATA_API_SECRET)
print("DATA_API_BASE_URL: ", DATA_API_BASE_URL)
print("TRADING_API_KEY: ", TRADING_API_KEY)
print("TRADING_API_SECRET: ", TRADING_API_SECRET)
print("TRADING_API_BASE_URL: ", TRADING_API_BASE_URL)

import alpaca_trade_api
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history
# from finrl.meta.PaperTradingAlpaca.alpaca_trade_api import PaperTradingAlpaca
from finrl.config import INDICATORS



# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# xgboost 2.1.1 requires nvidia-nccl-cu12; platform_system == "Linux" and platform_machine != "aarch64", which is not installed.
# cudf-cu12 24.4.1 requires pandas<2.2.2dev0,>=2.0, but you have pandas 2.2.2 which is incompatible.
# google-colab 1.0.0 requires pandas==2.1.4, but you have pandas 2.2.2 which is incompatible.
# osqp 0.6.7.post0 requires scipy!=1.12.0,>=0.13.2, but you have scipy 1.12.0 which is incompatible.


# pip install finrl.meta.paper_trading.alpaca
# pip install finrl.meta.paper_trading.common

# Import Dow Jones 30 Symbols
from finrl.config_tickers import DOW_30_TICKER

ticker_list = DOW_30_TICKER
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

train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    ALPACA_API_KEY=DATA_API_KEY,
    ALPACA_API_SECRET =DATA_API_SECRET,
    ALPACA_API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl",  # current_working_dir
    break_step=1e5,
)

account_value_erl = test(
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    ALPACA_API_KEY=DATA_API_KEY,
    ALPACA_API_SECRET =DATA_API_SECRET,
    ALPACA_API_BASE_URL=DATA_API_BASE_URL,
    cwd="./papertrading_erl",
    net_dimension=ERL_PARAMS["net_dimension"],
)

train(
    start_date=TRAINFULL_START_DATE,  # After tuning well, retrain on the training and testing sets
    end_date=TRAINFULL_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    ALPACA_API_KEY=DATA_API_KEY,
    ALPACA_API_SECRET =DATA_API_SECRET,
    ALPACA_API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl_retrain",
    break_step=2e5,
)

action_dim = len(DOW_30_TICKER)
state_dim = (
    1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
)  # Calculate the DRL state dimension manually for paper trading. amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim

paper_trading_erl = PaperTradingAlpaca(
    ticker_list=DOW_30_TICKER,
    time_interval="1Min",
    drl_lib="elegantrl",
    agent="ppo",
    cwd="./papertrading_erl_retrain",
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    ALPACA_API_KEY=TRADING_API_KEY,
    ALPACA_API_SECRET =TRADING_API_SECRET,
    ALPACA_API_BASE_URL=TRADING_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=1e2,
)

paper_trading_erl.run()

# Check Portfolio Performance
# ## Get cumulative return
df_erl, cumu_erl = alpaca_history(
    key=DATA_API_KEY,
    secret=DATA_API_SECRET,
    url=DATA_API_BASE_URL,
    start="2022-09-01",  # must be within 1 month
    end="2022-09-12",
)  # change the date if error occurs

df_djia, cumu_djia = DIA_history(start="2022-09-01")
returns_erl = cumu_erl - 1
returns_dia = cumu_djia - 1
returns_dia = returns_dia[: returns_erl.shape[0]]

# plot and save
import matplotlib.pyplot as plt

plt.figure(dpi=1000)
plt.grid()
plt.grid(which="minor", axis="y")
plt.title("Stock Trading (Paper trading)", fontsize=20)
plt.plot(returns_erl, label="ElegantRL Agent", color="red")
# plt.plot(returns_sb3, label = 'Stable-Baselines3 Agent', color = 'blue' )
# plt.plot(returns_rllib, label = 'RLlib Agent', color = 'green')
plt.plot(returns_dia, label="DJIA", color="grey")
plt.ylabel("Return", fontsize=16)
plt.xlabel("Year 2021", fontsize=16)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(78))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
ax.xaxis.set_major_formatter(
    ticker.FixedFormatter(["", "10-19", "", "10-20", "", "10-21", "", "10-22"])
)
plt.legend(fontsize=10.5)
plt.savefig("papertrading_stock.png")
