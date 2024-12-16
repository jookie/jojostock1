# /content/papertrading_erl/actor.pth
# /content/papertrading_erl_retrain/actor.pth

# Integrating Lumibot with A2C (Advantage Actor-Critic), einforcement learning algorithm, involves leveraging machine learning libraries alongside Lumibot’s trading framework.
# technical analysis strategies such as PPO (Percentage Price Oscillator) A2C (Advantage Actor-Critic). The PPO is a momentum indicator commonly used in financial trading strategies.
# Lumibot, a Python library for creating and backtesting trading strategies, can be integrated with technical analysis strategies such as PPO (Percentage Price Oscillator). The PPO is a momentum indicator commonly used in financial trading strategies.

from __future__ import annotations
import sys ; sys.path.append("~/lib/rl")
import streamlit as st
import warnings ; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from lib.rl.config_tickers import DOW_30_TICKER
from lib.rl.meta.preprocessor.yahoodownloader import YahooDownloader
from lib.rl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from lib.rl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.rl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from lib.rl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from lib.utility.jprint import jprint
# import alpaca.trading.enums
from lib.rl.main import check_and_make_directories
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
    OrderType,
    OrderSide,
    TimeInForce,
)

# st.set_page_config(page_title="Stock Training", page_icon="📹")
st.button("Re-run")
# st.markdown("# Stock Training")
# st.sidebar.header("Stock Testing")
st.write(
    """This app shows how you can use Streamlit to build cool animations.
It displays an animated fractal based on the the Julia Set. Use the slider
to tune different parameters."""
)
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])


TRAIN_START_DATE = '2009-04-01'
TRAIN_END_DATE = '2021-01-01'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2022-06-01'
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL

tic = DOW_30_TICKER
tic = [DOW_30_TICKER[0]]

st.write("jojo".join(str(ticker) for ticker in tic))
# if tic = ["AXP"] has 1 row it work , but if tic = [
#     "AXP",
#     "AMGN"
#     ] or greater than 1 I get the message below :

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TEST_END_DATE,
                     ticker_list = tic).fetch_data()

df.sort_values(['date','tic']).head()
print(len(df.tic.unique()))
print(df.tic.value_counts())
print(df.head())
print(df.tail())
print(df.shape)

st.write(len(df.tic.unique()))
# st.write(df.tic.value_counts())
# st.write(df.head())
# st.write(df.tail())
st.write(df.shape)

# INDICATORS = ['macd',
#                'rsi_30',
#                'cci_30',
#                'dx_30']
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf,0)

print(processed.sample(5))
st.write(processed.sample(5))

stock_dimension = len(processed.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
st.write(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
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
    "print_verbosity":5
    
}

rebalance_window = 63 #63 # rebalance_window is the number of days to retrain the model
validation_window = 63 #63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

ensemble_agent = DRLEnsembleAgent(df=processed,
                 train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
                 val_test_period=(TEST_START_DATE,TEST_END_DATE),
                 rebalance_window=rebalance_window, 
                 validation_window=validation_window, 
                 **env_kwargs)

A2C_model_kwargs = {
                    'n_steps': 5,
                    'ent_coef': 0.005,
                    'learning_rate': 0.0007
                    }

PPO_model_kwargs = {
                    "ent_coef":0.01,
                    "n_steps": 2, #2048
                    "learning_rate": 0.00025,
                    "batch_size": 128
                    }

DDPG_model_kwargs = {
                      #"action_noise":"ornstein_uhlenbeck",
                      "buffer_size": 1, #10_000
                      "learning_rate": 0.0005,
                      "batch_size": 64
                    }
SAC_model_kwargs = {
                      "buffer_size": 1, #10_000
                      "learning_rate": 0.0005,
                      "batch_size": 64}
TD3_model_kwargs = {
                      "buffer_size": 1, #10_000
                      "learning_rate": 0.0005,
                      "batch_size": 64,
}

timesteps_dict = {
    'a2c': 10,  # Example value, adjust as needed
    'ppo': 10,
    'ddpg': 10,
    'sac' : 10,
    'td3' : 10
}

df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs,
    PPO_model_kwargs,
    DDPG_model_kwargs,
    SAC_model_kwargs,
    TD3_model_kwargs,
    timesteps_dict
)

print(df_summary)
st.write(df_summary)

unique_trade_date = processed[(processed.date > TEST_START_DATE)&(processed.date <= TEST_END_DATE)].date.unique()
df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

df_account_value=pd.DataFrame()
for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
    temp = pd.read_csv(RESULTS_DIR + '/account_value_trade_{}_{}.csv'.format('ensemble',i))
    # temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
    df_account_value = df_account_value._append(temp,ignore_index=True)
sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
print('Sharpe Ratio: ',sharpe)
st.write('Sharpe Ratio: ',sharpe)


df_account_value=df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

df_account_value.head()
df_account_value.account_value.plot()
st.line_chart(df_account_value['account_value'])

jprint("==============Get Backtest Outcome===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)

#baseline stats
print("==============Get Baseline Stats===========")
st.write("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

stats = backtest_stats(baseline_df, value_col_name = 'close')

print("==============Compare to DJIA===========")
st.write("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value, 
              baseline_ticker = '^DJI', 
              baseline_start = df_account_value.loc[0,'date'],
              baseline_end = df_account_value.loc[len(df_account_value)-1,'date'])
st.line_chart(df_account_value['account_value'])

# /Users/dovpeles/jojobot1/jojostock1/venv/lib/python3.12/site-packages/alpaca/trading/enums.py
# from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
# from alpaca_py.alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import alpaca_trade_api as tradeapi

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET , ALPACA_API_BASE_URL, api_version='v2')
try:
    account = api.get_account()
    print(f"Account status: {account.status}")
    st.write(f"Account status: {account.status}")
    symbol = 'CAT'
    qty = 1  # Quantity to buy
    if (sharpe > - 94):
        buy_order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            type= OrderType.MARKET)
        print(f"Buy order submitted: {buy_order}")
        st.write(f"Buy order submitted: {buy_order}")
    else:
        print('no trades for today')
        st.write('no trades for today')
except Exception as e:
    print(f"An error occurred: {e}")
    st.write(f"An error occurred: {e}")


