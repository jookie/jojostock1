# http://localhost:8888/lab/tree/jojostock1/app.py
from __future__ import annotations
import os
import streamlit as st
import sys
import itertools
import matplotlib.pyplot as plt
sys.path.append('../lib/rl')
from lib.utility.jprint import jprint


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
)

from lib.rl.meta.preprocessor.yahoodownloader import YahooDownloader
from lib.rl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from lib.rl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.rl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from lib.rl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from lib.rl.main import check_and_make_directories
from lib.rl import config_tickers

import warnings
warnings.filterwarnings("ignore")
API_KEY = "PKVD6WOSPEMKS0UI6A3K"
API_SECRET = "BxT64PIQtDBb*tnW"
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'

custom_css = """
<style>
body {
background-color: black; /* Background color (black) */
font-family: "Times New Roman", Times, serif; /* Font family (Times New Roman) */
color: white; /* Text color (white) */
line-height: 1.6; /* Line height for readability */
}

h1 {
color: #3498db; /* Heading color (light blue) */
}

h2 {
color: #e74c3c; /* Subheading color (red) */
}

p {
margin: 10px 0; /* Margin for paragraphs */
}

</style>
"""

# Representing an upward movement
up_candle = "📈"  # \U0001F4C8
down_candle = ""  # \U0001F4C9
# Set page title and configure layout
st.set_page_config(
      page_title="Stock prediction Trainig", 
      layout="wide",
      page_icon= "📈"
      )
# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)
#page title and subtitle
st.title(" 📈 Stock Price Ptediction Training 📉")
st.markdown("Trainig to predict stock price movements for a given stock ticker symbol and its foundamental ratios")
finviz_url = "https://finviz.com/quote.ashx?t="
example_ticker_symbols = [
"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
"JPM", "NFLX", "FB", "BRK.B", "V",
"NVDA", "DIS", "BA", "IBM", "GE",
"PG", "JNJ", "KO", "MCD", "T",
"ADBE", "CRM", "INTC", "ORCL", "HD"
]
# Use a selectbox to allow users to choose from example ticker symbols
ticker = st.selectbox("Select a stock ticker symbol or enter your own:", example_ticker_symbols)
# if ticker:
      #Fetching stock price data
def main():
  import pandas as pd
  dir = [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
  check_and_make_directories( dir )
  
  jprint("app.py: Directory Paths:   ",  "   //".join(dir),  '##')
  """app.py: Waiting data collection From Yahoo downloader ..."""
  df = YahooDownloader(start_date = TRAIN_START_DATE,
                      end_date = TRADE_END_DATE,
                      ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
#  df.shape
  df.sort_values(['date','tic'],ignore_index=True).head()
  fe = FeatureEngineer(
                      use_technical_indicator=True,
                      tech_indicator_list = INDICATORS,
                      use_vix=True,
                      use_turbulence=True,
                      user_defined_feature = False)

  processed = fe.preprocess_data(df)
  list_ticker = processed["tic"].unique().tolist()
  list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
  combination = list(itertools.product(list_date,list_ticker))
  processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
  processed_full = processed_full[processed_full['date'].isin(processed['date'])]
  processed_full = processed_full.sort_values(['date','tic'])
  processed_full = processed_full.fillna(0)
  processed_full.sort_values(['date','tic'],ignore_index=True).head(10)
  train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
  trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)

  st.write(train.head()) ; 
  st.write(train.tail())
  
  mvo_df = processed_full.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]
  stock_dimension = len(train.tic.unique())
  state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
  st.write(f"Trained num of Symboles: {stock_dimension}, State Total Space: {state_space}")
  buy_cost_list = sell_cost_list = [0.001] * stock_dimension
  num_stock_shares = [0] * stock_dimension
  env_kwargs = {
      "hmax": 100,
      "initial_amount": 1000000,
      "num_stock_shares": num_stock_shares,
      "buy_cost_pct": buy_cost_list,
      "sell_cost_pct": sell_cost_list,
      "state_space": state_space,
      "stock_dim": stock_dimension,
      "tech_indicator_list": INDICATORS,
      "action_space": stock_dimension,
      "reward_scaling": 1e-4
  }
  e_train_gym = StockTradingEnv(df = train, **env_kwargs)
  env_train, _ = e_train_gym.get_sb_env()
  
  # st.write(type(env_train))
  
  agent = DRLAgent(env = env_train)
  if_using_a2c = True
  if_using_ddpg = True
  if_using_ppo = True
  if_using_td3 = True
  if_using_sac = True

  def predict_with_models(models, environment):
      """
      Perform predictions using multiple trained models in the specified environment.

      Parameters:
      - models: A dictionary of trained models with names as keys.
      - environment: The trading environment to be used for predictions.

      Returns:
      - results: A dictionary containing DataFrames of account values and actions for each model.
      """
      results = {}

      for model_name, trained_model in models.items():
          df_account_value, df_actions = DRLAgent.DRL_prediction(
              model=trained_model,
              environment=environment
          )
          results[model_name] = {
              "account_value": df_account_value,
              "actions": df_actions
          }
      return results
      
  def train_agent(agent, model_name = "a2c", total_timesteps=50000):
      """
      Train a model with the provided agent and model_name and total_timesteps 
      """
      # Get the model for A2C if applicable
      __cached__model_ = agent.get_model(model_name)
      # Set up logger
      _tmp_path = RESULTS_DIR + '/' + model_name
      _new_logger = configure(_tmp_path, ["stdout", "csv", "tensorboard"])
      # Set the new logger
      __cached__model_.set_logger(_new_logger)
          
      # Train the model
      _trained = agent.train_model(
      model=__cached__model_, 
      tb_log_name=model_name,
      total_timesteps=total_timesteps
      )
      return _trained

  trained_a2c = train_agent(agent, "a2c", 5) if if_using_a2c else None
  trained_ddpg = train_agent(agent, "ddpg", 5) if if_using_ddpg else None
  trained_ppo = train_agent(agent, "ppo", 5) if if_using_ppo else None  
  trained_td3 = train_agent(agent, "td3", 5) if if_using_td3 else None
  trained_sac = train_agent(agent, "sac", 5) if if_using_sac else None
  
  data_risk_indicator = processed_full[(processed_full.date<TRAIN_END_DATE) & (processed_full.date>=TRAIN_START_DATE)]
  insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])
  
  st.write(f"Vix Indicator: {insample_risk_indicator.vix.quantile(0.996)}")
  st.write(insample_risk_indicator.vix.describe())
  st.write(insample_risk_indicator.turbulence.describe())
  st.write(insample_risk_indicator.turbulence.quantile(0.996))

  e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold =70,  risk_indicator_col='vix', **env_kwargs)
  env_trade, obs_trade = e_trade_gym.get_sb_env()

  st.write(trade.head())
  st.write(trade.tail())
  # Example usage:
  models = {
      "a2c": trained_a2c,
      "ddpg": trained_ddpg,
      "ppo": trained_ppo,
      "td3": trained_td3,
      "sac": trained_sac
  }
  results = predict_with_models(models, e_trade_gym)
  # Access results for each model
  df_account_value_a2c = results["a2c"]["account_value"]
  df_actions_a2c = results["a2c"]["actions"]

  df_account_value_ddpg = results["ddpg"]["account_value"]
  df_actions_ddpg = results["ddpg"]["actions"]

  df_account_value_ppo = results["ppo"]["account_value"]
  df_actions_ppo = results["ppo"]["actions"]

  df_account_value_td3 = results["td3"]["account_value"]
  df_actions_td3 = results["td3"]["actions"]

  df_account_value_sac = results["sac"]["account_value"]
  df_actions_sac = results["sac"]["actions"]


  st.write(df_account_value_a2c.shape)
  st.write(df_account_value_a2c.head())
  st.write(df_account_value_a2c.tail())

  fst = mvo_df
  fst = fst.iloc[0*29:0*29+29, :]
  tic = fst['tic'].tolist()

  mvo = pd.DataFrame()

  for k in range(len(tic)):
    mvo[tic[k]] = 0

  for i in range(mvo_df.shape[0]//29):
    n = mvo_df
    n = n.iloc[i*29:i*29+29, :]
    date = n['date'][i*29]
    mvo.loc[date] = n['close'].tolist()

#   mvo.shape[0]  

  from scipy import optimize 
  from scipy.optimize import linprog

  #function obtains maximal return portfolio using linear programming

  def MaximizeReturns(MeanReturns, PortfolioSize):
      
    #dependencies
    
      
    c = (np.multiply(-1, MeanReturns))
    A = np.ones([PortfolioSize,1]).T
    b=[1]
    res = linprog(c, A_ub = A, b_ub = b, bounds = (0,1), method = 'simplex') 
      
    return res

  def MinimizeRisk(CovarReturns, PortfolioSize):
      
    def f(x, CovarReturns):
      func = np.matmul(np.matmul(x, CovarReturns), x.T) 
      return func

    def constraintEq(x):
      A=np.ones(x.shape)
      b=1
      constraintVal = np.matmul(A,x.T)-b 
      return constraintVal
      
    xinit=np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun':constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])

    opt = optimize.minimize (f, x0 = xinit, args = (CovarReturns),  bounds = bnds, \
                              constraints = cons, tol = 10**-3)
      
    return opt

  def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
      
    def  f(x,CovarReturns):
          
      func = np.matmul(np.matmul(x,CovarReturns ), x.T)
      return func

    def constraintEq(x):
      AEq=np.ones(x.shape)
      bEq=1
      EqconstraintVal = np.matmul(AEq,x.T)-bEq 
      return EqconstraintVal
      
    def constraintIneq(x, MeanReturns, R):
      AIneq = np.array(MeanReturns)
      bIneq = R
      IneqconstraintVal = np.matmul(AIneq,x.T) - bIneq
      return IneqconstraintVal
      

    xinit=np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun':constraintEq},
            {'type':'ineq', 'fun':constraintIneq, 'args':(MeanReturns,R) })
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])

    opt = optimize.minimize (f, args = (CovarReturns), method ='trust-constr',  \
                  x0 = xinit,   bounds = bnds, constraints = cons, tol = 10**-3)
      
    return opt

  def StockReturnsComputing(StockPrice, Rows, Columns): 
    import numpy as np 
    StockReturn = np.zeros([Rows-1, Columns]) 
    for j in range(Columns):        # j: Assets 
      for i in range(Rows-1):     # i: Daily Prices 
        StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100 
        
    return StockReturn

  # Obtain optimal portfolio sets that maximize return and minimize risk

  #Dependencies
  import numpy as np
  import pandas as pd


  #input k-portfolio 1 dataset comprising 15 stocks
  # StockFileName = './DJIA_Apr112014_Apr112019_kpf1.csv'

  Rows = 1259  #excluding header
  Columns = 15  #excluding date
  portfolioSize = 29 #set portfolio size

  #read stock prices in a dataframe
  # df = pd.read_csv(StockFileName,  nrows= Rows)

  #extract asset labels
  # assetLabels = df.columns[1:Columns+1].tolist()
  # st.write(assetLabels)

  #extract asset prices
  # StockData = df.iloc[0:, 1:]
  StockData = mvo.head(mvo.shape[0]-336)
  st.write(StockData)
  TradeData = mvo.tail(336)
  st.write(df.head())
  TradeData.to_numpy()

  #compute asset returns
  arStockPrices = np.asarray(StockData)
  [Rows, Cols]=arStockPrices.shape
  arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)


  #compute mean returns and variance covariance matrix of returns
  meanReturns = np.mean(arReturns, axis = 0)
  covReturns = np.cov(arReturns, rowvar=False)
  
  #set precision for printing results
  np.set_printoptions(precision=3, suppress = True)

  #display mean returns and variance-covariance matrix of returns
  st.write('Mean returns of assets in k-portfolio 1\n', meanReturns)
  st.write('Variance-Covariance matrix of returns\n', covReturns)

  from pypfopt.efficient_frontier import EfficientFrontier

  ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
  raw_weights_mean = ef_mean.max_sharpe()
  cleaned_weights_mean = ef_mean.clean_weights()
  mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
  st.write(mvo_weights)

  st.write(StockData.tail(1))

  Portfolio_Assets = TradeData # Initial_Portfolio
  MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
#   MVO_result

  df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
  df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
  df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
  df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
  df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])
  
  def mkdirDataDf(fn):
    folder_path = os.path.join("data", "df")
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, fn )
    return file_path
  
  df_account_value_a2c.to_csv( mkdirDataDf( "df_account_value_a2c.csv"))
  #baseline stats
  st.write("==============Get Baseline Stats===========")
  df_dji_ = get_baseline(
          ticker="^DJI", 
          start = TRADE_START_DATE,
          end = TRADE_END_DATE)
  stats = backtest_stats(df_dji_, value_col_name = 'close')
  df_dji = pd.DataFrame()
  df_dji['date'] = df_account_value_a2c['date']
  df_dji['account_value'] = df_dji_['close'] / df_dji_['close'][0] * env_kwargs["initial_amount"]
  df_dji.to_csv(mkdirDataDf("df_dji.csv"))
  df_dji = df_dji.set_index(df_dji.columns[0])
  df_dji.to_csv(mkdirDataDf("df_dji+.csv"))


  result = pd.merge(df_result_a2c, df_result_ddpg, left_index=True, right_index=True, suffixes=('_a2c', '_ddpg'))
  result = pd.merge(result, df_result_td3, left_index=True, right_index=True, suffixes=('', '_td3'))
  result = pd.merge(result, df_result_ppo, left_index=True, right_index=True, suffixes=('', '_ppo'))
  result = pd.merge(result, df_result_sac, left_index=True, right_index=True, suffixes=('', '_sac'))
  result = pd.merge(result, MVO_result, left_index=True, right_index=True, suffixes=('', '_mvo'))
  result = pd.merge(result, df_dji, left_index=True, right_index=True, suffixes=('', '_dji'))
  result.columns = ['a2c', 'ddpg', 'td3', 'ppo', 'sac', 'mean var', 'dji']

  st.write("result: ", result)
  result.to_csv(mkdirDataDf("result.csv"))

  plt.rcParams["figure.figsize"] = (15,5)
  fig, ax = plt.subplots()
  result.plot(ax=ax)
  st.pyplot(fig)





if __name__ == "__main__":
    main()