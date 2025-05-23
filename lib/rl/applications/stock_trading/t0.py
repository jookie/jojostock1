# https://zhuanlan.zhihu.com/p/616799055



from __future__ import annotations
# Add project root to Python path
import sys
import os
path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(path_to_add)
print(f"Added path: {path_to_add}")
# print(f"sys.path: {sys.path}")

def main():
    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    
    # matplotlib.use('Agg')
    import datetime
  
    from finrl.config import INDICATORS
    # from finrl.meta.preprocessor.alpacadownloader import AlpacaDownloader
    
    from finrl.meta.preprocessor.alpacadownloader import FeatureEngineer
    from finrl.meta.data_processor import DataProcessor
    import pandas as pd
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.config import TEST_END_DATE    # "2021-11-01"
    from finrl.config import TRAIN_START_DATE # "2014-01-06" 
    from finrl.config_tickers import DOW_30_TICKER
    from alpaca.data.timeframe import TimeFrame
    
    from finrl.main import check_and_make_directories
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )
    
    
    # Define parameters
    ticker_list = ["AAPL", "MSFT"]
    start_date = TRAIN_START_DATE #"2021-01-01"
    end_date = TEST_END_DATE      # "2021-12-31"
    time_interval = TimeFrame.Minute # Supported: 1Min, 5Min, 15Min, 1H, 1D
 
    print(ticker_list)
    df = AlpacaDownloader().fetch_data(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=DOW_30_TICKER, time_interval = time_interval)

    # print(df) 
    # df.sort_values(["date", "tic"]).head()
    
    # fe = FeatureEngineer(
    #     use_technical_indicator=True,
    #     tech_indicator_list=INDICATORS,
    #     use_turbulence=True,
    #     user_defined_feature=False,
    # )

    # processed = fe.preprocess_data(df)
    # processed = processed.copy()
    # processed = processed.fillna(0)
    # processed = processed.replace(np.inf, 0)

    # stock_dimension = len(processed.tic.unique())
    # state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    
    # print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

if __name__ == "__main__":
    raise SystemExit(main())

