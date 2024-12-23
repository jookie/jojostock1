
from datetime import datetime , timedelta 
import streamlit as st
from lumibot.credentials import IS_BACKTESTING
from lumibot.backtesting.yahoo_backtesting import  YahooDataBacktesting
from lumibot.example_strategies.stock_bracket import StockBracket
from lumibot.example_strategies.drift_rebalancer import DriftRebalancer
from lumibot.example_strategies.ccxt_backtesting_example import CcxtBacktesting
from lumibot.entities import Asset, Order
from lumibot.backtesting import CcxtBacktesting
if  __name__ == "__main__":

    base_symbol = "ETH"
    quote_symbol = "USDT"
    start_date = datetime(2023,2,11)
    end_date = datetime(2024,2,12)
    asset = (Asset(symbol=base_symbol, asset_type="crypto"),
            Asset(symbol=quote_symbol, asset_type="crypto"))

    exchange_id = "kraken"  #"kucoin" #"bybit" #"okx" #"bitmex" # "binance"


    # CcxtBacktesting default data download limit is 50,000
    # If you want to change the maximum data download limit, you can do so by using 'max_data_download_limit'.
    kwargs = {
        # "max_data_download_limit":10000, # optional
        "exchange_id":exchange_id,
    }
    CcxtBacktesting.MIN_TIMESTEP = "day"
    results, strat_obj = CcxtBacktestingExampleStrategy.run_backtest(
        CcxtBacktesting,
        start_date,
        end_date,
        benchmark_asset=f"{base_symbol}/{quote_symbol}",
        quote_asset=Asset(symbol=quote_symbol, asset_type="crypto"),
        parameters={
                "asset":asset,
                "cash_at_risk":.25,
                "window":21,},
        **kwargs,
    )
# if __name__ == "__main__":
#     backtesting_start = datetime(2023, 1, 2)
#     backtesting_end  = datetime(2024, 10, 31)

#     st.write(f"Backtesting: Start {backtesting_start} End {backtesting_end}")
#     IS_BACKTESTING = True
#     if not IS_BACKTESTING:
#         print("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")
#         st.write("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")        
#         exit()
#     else:
#         parameters = {
#             "market": "NYSE",
#             "sleeptime": "1D",
#             "drift_threshold": "0.05",
#             "acceptable_slippage": "0.005",  # 50 BPS
#             "fill_sleeptime": 15,
#             "target_weights": {
#                 "SPY": "0.60",
#                 "TLT": "0.40"
#             },
#             "shorting": False
#         }

#         results = DriftRebalancer.backtest(
#             # StockBracket
#             YahooDataBacktesting,
#             backtesting_start,
#             backtesting_end,
#             benchmark_asset="SPY",
#             # Others,
#             parameters=parameters,
#             show_plot=False,
#             show_tearsheet=False,
#             save_tearsheet=False,
#             show_indicators=False,
#             save_logfile=False,
#             # show_progress_bar=False,
#             # quiet_logs=False
#         )

#         print(results)
#         st.write(results)
    
#         # results = StockBracket.backtest(
#         #     YahooDataBacktesting,
#         #     backtesting_start,
#         #     backtesting_end,
#         #     benchmark_asset="SPY",
#         # )    
    
