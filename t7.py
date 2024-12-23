
from datetime import datetime , timedelta 
import streamlit as st
from lumibot.credentials import IS_BACKTESTING
from lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from lumibot.example_strategies.stock_bracket import StockBracket
from lumibot.example_strategies.drift_rebalancer import DriftRebalancer

if __name__ == "__main__":
    backtesting_start = datetime(2023, 1, 2)
    backtesting_end  = datetime(2024, 10, 31)

    st.write(f"Backtesting: Start {backtesting_start} End {backtesting_end}")
    IS_BACKTESTING = True
    if not IS_BACKTESTING:
        print("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")
        st.write("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")        
        exit()
    else:
        parameters = {
            "market": "NYSE",
            "sleeptime": "1D",
            "drift_threshold": "0.05",
            "acceptable_slippage": "0.005",  # 50 BPS
            "fill_sleeptime": 15,
            "target_weights": {
                "SPY": "0.60",
                "TLT": "0.40"
            },
            "shorting": False
        }

        results = DriftRebalancer.backtest(
            # StockBracket
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            benchmark_asset="SPY",
            # Others,
            parameters=parameters,
            show_plot=False,
            show_tearsheet=False,
            save_tearsheet=False,
            show_indicators=False,
            save_logfile=False,
            # show_progress_bar=False,
            # quiet_logs=False
        )

        print(results)
        st.write(results)
    
        # results = StockBracket.backtest(
        #     YahooDataBacktesting,
        #     backtesting_start,
        #     backtesting_end,
        #     benchmark_asset="SPY",
        # )    
    
