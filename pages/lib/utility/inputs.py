
# https://github.com/mkhorasani/Streamlit-Authenticator
# https://streamlit.io/componen
# https://okld-gallery.streamlit.app/?p=elements
# https://extras.streamlit.app/
# https://arnaudmiribel.github.io/streamlit-extras/extras/sandbox/
# https://github.com/imdreamer2018/streamlit-date-picker

import streamlit as st
from lib.utility.jprint import jprint

from lib.rl.config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TRADE_START_DATE, TRADE_END_DATE

from lib.rl.meta.preprocessor.yahoodownloader import YahooDownloader
from lib.rl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from lib.rl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.rl.agents.stablebaselines3.models import DRLAgent
from lib.rl.config_tickers import index_dict, sector_dict, usa_dict, SP_500_TICKER
from stable_baselines3.common.logger import configure

from datetime import datetime
import os
import itertools
from lib.rl.config import (
      DATA_FRAME_DIR,
      INDICATORS,
      RESULTS_DIR,
)
def setFirstPageTitle() : 
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
    st.markdown(custom_css, unsafe_allow_html=True)
    set_inputs99()  
    # st.title(" 📈 Stock Price Ptediction Training")
    # st.markdown("Trainig to predict stock price movements for a given stock ticker symbol and its foundamental ratios") 
    
def set_inputs99():

    st.title("Dynamic Financial Reinforcement Learning")
    st.write("""
    This application simulates a dynamic dataset-driven financial reinforcement learning model, 
    which uses a rolling window technique to incrementally update the training and testing sets based on real-time market data.
    The dataset is divided into training and testing segments, which adjust every W days to keep the model updated.
    """)

class WorkflowScheduler:
    def __init__(self):
        # Define labels and date ranges for different workflow modes
        self.labels = {
            "Train": (TRAIN_START_DATE, TRAIN_END_DATE),
            "Test": (TEST_START_DATE, TEST_END_DATE),
            "Trade": (TRADE_START_DATE, TRADE_END_DATE),
        }
        self.train_start_date = self.labels["Train"][0]
        self.train_end_date = self.labels["Train"][1]
        self.test_start_date = self.labels["Test"][0]
        self.test_end_date = self.labels["Test"][1]
        self.trade_start_date = self.labels["Trade"][0]
        self.trade_end_date = self.labels["Trade"][1]
    
    def display_sidebar(self):
        
        st.sidebar.header("Rolling Window")
        df = "%Y-%m-%d"
        # Display headers
        col1, col2, col3 = st.sidebar.columns([1, 2, 2]) 
                    
        # Center-align column headers
        col1.markdown("<div style='text-align: center; font-weight: bold;'>Mode</div>", unsafe_allow_html=True)
        col2.markdown("<div style='text-align: center; font-weight: bold;'>Start</div>", unsafe_allow_html=True)
        col3.markdown("<div style='text-align: center; font-weight: bold;'>End</div>", unsafe_allow_html=True)

        for label, (start, end) in self.labels.items():
            # Convert start and end dates to datetime objects
            start_date = datetime.strptime(start, df).date()
            end_date   = datetime.strptime(end, df).date()
            with col1:
                st.write(label)
            with col2:
                start_date = st.date_input(f"{label}_start", value=datetime.strptime(start, df).date(), label_visibility="collapsed")
            with col3:
                end_date   = st.date_input(f"{label}_end", value=datetime.strptime(end, df).date(), label_visibility="collapsed")
            # Save the selected dates as attributes of self
            setattr(self, f"{label.lower()}_start_date", start_date.strftime(df))
            setattr(self, f"{label.lower()}_end_date"  , end_date.strftime(df))
 
def get_full_path(fn):
    file_path = os.path.join(DATA_FRAME_DIR, fn )
    return file_path
 
def set_yahoo_data_frame(ticker_ls, wf) :
      
  """app.py: Waiting data collection From Yahoo downloader ..."""
  df = YahooDownloader(start_date  = 
  wf.train_start_date,
  end_date = wf.trade_end_date,
  ticker_list = ticker_ls).fetch_data()
  
  df.sort_values(['date','tic'],ignore_index=True).head()
  
  fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)
  st.write(df.shape)
  processed = fe.preprocess_data(df)

  import pandas as pd
  
  list_ticker = processed["tic"].unique().tolist()
  list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
  combination = list(itertools.product(list_date,list_ticker))
  processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
  processed_full = processed_full[processed_full['date'].isin(processed['date'])]
  processed_full = processed_full.sort_values(['date','tic'])
  processed_full = processed_full.fillna(0)
  processed_full.sort_values(['date','tic'],ignore_index=True).head(10)
  
  train = data_split(processed_full, wf.train_start_date, wf.train_end_date)
  trade = data_split(processed_full, wf.trade_start_date, wf.trade_end_date)
  st.write(train.head()) ; st.write(train.tail())

  mvo_df = processed_full.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]
  stock_dimension = len(train.tic.unique())
  state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
  st.write(f"Trained num of Symboles: {stock_dimension}, State Total Space: {state_space}")  
  
  from collections import namedtuple
  Result = namedtuple("Result","df, processed_full, train, trade, mvo_df, stock_dimension, state_space")

 
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
  agent = DRLAgent(env = env_train)
  if_using_a2c = True
  if_using_ddpg = True
  if_using_ppo = True
  if_using_td3 = True
  if_using_sac = True  
  
  trained_a2c = train_agent(agent, "a2c", 5) if if_using_a2c else None
  trained_ddpg = train_agent(agent, "ddpg", 5) if if_using_ddpg else None
  trained_ppo = train_agent(agent, "ppo", 5) if if_using_ppo else None  
  trained_td3 = train_agent(agent, "td3", 5) if if_using_td3 else None
  trained_sac = train_agent(agent, "sac", 5) if if_using_sac else None
  models = {
      "a2c": trained_a2c,
      "ddpg": trained_ddpg,
      "ppo": trained_ppo,
      "td3": trained_td3,
      "sac": trained_sac
  }
  return mvo_df, env_kwargs, trade, processed_full , models 
   
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

def GetTickerList():
    """
    Generate a list of tickers based on user selection (Index, Sector, or NYSE) in a Streamlit app.

    Parameters:
    - index_dict: Dictionary of indexes and their respective tickers.
    - sector_dict: Dictionary of sectors and their respective tickers.
    - usa_dict: Dictionary of NYSE-specific categories and tickers.
    - SP_500_TICKER: List of tickers for the S&P 500.

    Returns:
    - final_ticker_list: List of selected tickers.
    """
    # Initialize session state for retaining previous selections
    st.session_state.setdefault("previous_index", list(index_dict.keys())[0])
    st.session_state.setdefault("previous_type", "Index")

    # Sidebar selection for type (Index, Sector, or NYSE)
    col1, col2 = st.sidebar.columns([1, 2])

    with col1:
        selection_type = st.radio(
            label="lbl1",
            options=["Index", "Sector", "NYSE"],
            horizontal=False,
            label_visibility="collapsed",
        )
        st.caption(f"Select {selection_type}:")
        options_dict = {
            "Index": index_dict,
            "Sector": sector_dict,
            "NYSE": usa_dict,
        }.get(selection_type, index_dict)

    with col2:
        # Dropdown to select a specific option based on the selection type
        selected_option = st.selectbox(
            f"Choose a {selection_type}",
            options=list(options_dict.keys()),
            label_visibility="collapsed",
        )

    # Determine default and options for multi-select
    if "first_symbol" not in st.session_state:
        st.session_state.first_symbol = SP_500_TICKER[0]
        default = SP_500_TICKER[1]
        options = SP_500_TICKER
    else:
        default = (
            SP_500_TICKER[1]
            if selection_type == "NYSE"
            else options_dict[selected_option]
        )
        options = options_dict[selected_option]

    # Multi-select widget for tickers
    selected_tickers = st.multiselect(
        label="Select Tickers",
        options=options,
        default=default,
        label_visibility="collapsed",
    )

    # Update session state if tickers are selected
    if selected_tickers:
        st.session_state.previous_index = selected_option
        st.session_state.previous_type = selection_type

    # Use previous selection as fallback if no tickers are selected
    final_ticker_list = (
        selected_tickers if selected_tickers else options_dict[st.session_state.previous_index]
    )

    return final_ticker_list
