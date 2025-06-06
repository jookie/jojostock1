# t1.py

# https://colab.research.google.com/github/mariko-sawada/FinRL_with_fundamental_data/blob/main/FinRL_stock_trading_fundamental.ipynb#scrollTo=f2wZgkQXh1jE

# https://github.com/search?q=FinRL&type=repositories&p=2

# https://medium.com/@mariko.sawada1/automated-stock-trading-with-deep-reinforcement-learning-and-financial-data-a63286ccbe2b

# https://github.com/mariko-sawada/FinRL_with_fundamental_data/blob/main/FinRL_stock_trading_fundamental.ipynb?source=post_page-----a63286ccbe2b---------------------------------------

# https://github.com/mariko-sawada/FinRL_with_fundamental_data

# AI4Finance-Foundation
# FinRL-Meta
# https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/examples/run_markowitz_portfolio_optimization.py

# AI4Finance-Foundation
# FinRobot
# https://github.com/AI4Finance-Foundation/FinRobot


# Get started with Trading API
# Browse More
# Trade using algorithms and bots with our easy-to-use API or web dashboard
# https://alpaca.markets/learn/fetch-historical-data

from __future__ import annotations
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
print(f"Added path: {project_root}")
import warnings
warnings.filterwarnings("ignore")
#+++++++++++++++++++++++++++++++++++++++++#
from finrl.config import INDICATORS
import numpy as np

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TEST_END_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,  #"2020-08-01"
    TEST_END_DATE, #"2021-10-01"
    TRADE_START_DATE, #"2021-11-01"
)

from alpaca.data.timeframe import TimeFrame
from finrl.meta.preprocessor.alpacadownloader import FeatureEngineer

ticker_list:list = ["AAPL", "MSFT"]
start_date:str = TRAIN_START_DATE #"2021-01-01"
end_date:str = TEST_END_DATE      # "2021-12-31"
# TODO TimeFrame Not used, should format 
time_interval = "1D" # Alapaca Support: 1Min, 5Min, 15Min, 1H, 1D
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False
)

df = fe.download_data(ticker_list = ticker_list, start_date = start_date, end_date = end_date, time_interval = time_interval)
print("Raw data:\n", df)

# Preprocess data
processed = fe.preprocess_data(df)
processed = processed.rename(columns={
    'timestamp': 'date',
})
# Handle missing values with interpolation and forward/backward fill
processed = processed.copy()
processed = processed.interpolate(method='linear', limit_direction='both')  # Interpolate numeric columns
processed = processed.ffill().bfill()  # Final fill for any remaining NaNs

# Check for infinite values and clip them
numeric_columns = processed.select_dtypes(include=[np.number]).columns
processed[numeric_columns] = processed[numeric_columns].clip(lower=-1e10, upper=1e10)

# Calculate state space
stock_dimension = len(processed.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

# Part 5. Design Environment
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5,
}

rebalance_window = 63  # rebalance_window is the number of days to retrain the model
validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
ensemble_agent = DRLEnsembleAgent(
    df=processed,
    train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
    val_test_period=(TEST_START_DATE, TEST_END_DATE),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)

A2C_model_kwargs = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0007}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

DDPG_model_kwargs = {
    # "action_noise":"ornstein_uhlenbeck",
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 64,
}

timesteps_dict = {"a2c": 10_000, "ppo": 10_000, "ddpg": 10_000}
import pandas as pd
df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
)

# unique_trade_date = processed[
#     (processed.date > TEST_START_DATE) & (processed.date <= TEST_END_DATE)
# ].date.unique()

# df_trade_date = pd.DataFrame({"datadate": unique_trade_date})

# df_account_value = pd.DataFrame()
# for i in range(
#     rebalance_window + validation_window,
#     len(unique_trade_date) + 1,
#     rebalance_window,
# ):
#     temp = pd.read_csv(
#         "results/account_value_trade_{}_{}.csv".format("ensemble", i)
#     )
#     df_account_value = df_account_value.append(temp, ignore_index=True)
# sharpe = (
#     (252**0.5)
#     * df_account_value.account_value.pct_change(1).mean()
#     / df_account_value.account_value.pct_change(1).std()
# )
# print("Sharpe Ratio: ", sharpe)
# df_account_value = df_account_value.join(
#     df_trade_date[validation_window:].reset_index(drop=True)
# )

# df_account_value.account_value.plot()

# import datetime

# print("==============Get Backtest Results===========")
# now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
# from finrl.plot import backtest_stats, get_baseline
# perf_stats_all = backtest_stats(account_value=df_account_value)
# perf_stats_all = pd.DataFrame(perf_stats_all)

# # baseline stats
# print("==============Get Baseline Stats===========")
# baseline_df = get_baseline(
#     ticker="^DJI",
#     start=df_account_value.loc[0, "date"],
#     end=df_account_value.loc[len(df_account_value) - 1, "date"],
# )

# stats = backtest_stats(baseline_df, value_col_name="close")

# print("==============Compare to DJIA===========")

# # S&P 500: ^GSPC
# # Dow Jones Index: ^DJI
# # NASDAQ 100: ^NDX
# from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
# backtest_plot(
#     df_account_value,
#     baseline_ticker="^DJI",
#     baseline_start=df_account_value.loc[0, "date"],
#     baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
# )



