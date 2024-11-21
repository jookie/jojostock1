
import streamlit as st
st.set_page_config(
    page_title="Stock prediction Trainig", 
    layout="wide",
    page_icon=':bar_chart:',
    )
from lib.utility.jprint import jprint
from lib.rl.config import (
      DATA_SAVE_DIR,
      TRAINED_MODEL_DIR,
      TENSORBOARD_LOG_DIR,
      DATA_FRAME_DIR,
      RESULTS_DIR,
)
from lib.rl.meta.preprocessor.yahoodownloader import YahooDownloader
from lib.rl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from lib.rl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.rl.agents.stablebaselines3.models import DRLAgent
from lib.rl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from lib.rl.main import check_and_make_directories
from lib.rl.config_tickers import index_dict
import matplotlib.pyplot as plt
import warnings ; warnings.filterwarnings("ignore")

from lib.utility.inputs import get_full_path, GetTickerList, set_yahoo_data_frame, predict_with_models

def main(ticker_list, _wf):
  import pandas as pd
  
  mvo_df, env_kwargs, trade, processed_full, models = set_yahoo_data_frame(ticker_list, _wf)
  
  def get_e_trade_gym_results():
    data_risk_indicator = processed_full[(processed_full.date<wf.train_end_date) & (processed_full.date>= wf.train_start_date)]
    insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])
    
    st.write(f"Vix Indicator: {insample_risk_indicator.vix.quantile(0.996)}")
    st.write(insample_risk_indicator.vix.describe())
    st.write(insample_risk_indicator.turbulence.describe())
    st.write(insample_risk_indicator.turbulence.quantile(0.996))

    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold =70,  risk_indicator_col='vix', **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    st.write(trade.head())
    st.write(trade.tail())
    
    results = predict_with_models(models, e_trade_gym)
    
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
    
    df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
    df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])
      
    result = pd.merge(df_result_a2c, df_result_ddpg, left_index=True, right_index=True, suffixes=('_a2c', '_ddpg'))
    result = pd.merge(result, df_result_td3, left_index=True, right_index=True, suffixes=('', '_td3'))
    result = pd.merge(result, df_result_ppo, left_index=True, right_index=True, suffixes=('', '_ppo'))
    result = pd.merge(result, df_result_sac, left_index=True, right_index=True, suffixes=('', '_sac'))  
    
    df_account_value_a2c.to_csv( get_full_path( "df_account_value_a2c.csv")) 
    return result, df_account_value_a2c
    
  
  result, df_account_value_a2c = get_e_trade_gym_results()
  
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
  # procDataFrame = pd.read_csv(StockFileName,  nrows= Rows)

  #extract asset labels
  # assetLabels = procDataFrame.columns[1:Columns+1].tolist()
  # st.write(assetLabels)

  #extract asset prices
  # StockData = procDataFrame.iloc[0:, 1:]
  StockData = mvo.head(mvo.shape[0]-336)
  st.write(StockData)
  TradeData = mvo.tail(336)
  
  # st.write(procDataFrame.head())
  
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
  
  st.write("==============Get Baseline Stats===========")
  df_dji_ = get_baseline(
          ticker="^DJI", 
          start = wf.train_start_date,
          end   = wf.train_end_date)
  stats = backtest_stats(df_dji_, value_col_name = 'close')
  df_dji = pd.DataFrame()
  df_dji['date'] = df_account_value_a2c['date']
  df_dji['account_value'] = df_dji_['close'] / df_dji_['close'][0] * env_kwargs["initial_amount"]
  df_dji.to_csv(get_full_path("df_dji.csv"))
  df_dji = df_dji.set_index(df_dji.columns[0])
  df_dji.to_csv(get_full_path("df_dji+.csv"))


  result = pd.merge(result, MVO_result, left_index=True, right_index=True, suffixes=('', '_mvo'))
  result = pd.merge(result, df_dji, left_index=True, right_index=True, suffixes=('', '_dji'))
  result.columns = ['a2c', 'ddpg', 'td3', 'ppo', 'sac', 'mean var', 'dji']

  st.write("result: ", result)
  result.to_csv(get_full_path("result.csv"))

  plt.rcParams["figure.figsize"] = (15,5)
  fig, ax = plt.subplots()
  result.plot(ax=ax)
  st.pyplot(fig)

if __name__ == "__main__":
  from lib.utility.inputs import WorkflowScheduler, setFirstPageTitle
  
  setFirstPageTitle()
  check_and_make_directories( [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR,    RESULTS_DIR, DATA_FRAME_DIR] )
  ticker_list = GetTickerList()
  wf = WorkflowScheduler()
  wf.display_sidebar()
  
  if st.button("Download Data per Ticket set"):
    main(ticker_list, wf)