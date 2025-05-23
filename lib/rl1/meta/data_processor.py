from __future__ import annotations
# from lib.utility.jprint import jprint
import numpy as np
import pandas as pd

from lib.rl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from lib.rl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from lib.rl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)
from alpaca.data.timeframe import TimeFrame
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
ALPACA_API_BASE_URL = 'https://paper-api.alpaca.markets'
from alpaca.data.historical import StockHistoricalDataClient

from alpaca.data.requests import StockBarsRequest
from alpaca.data.models.bars import BarSet
import alpaca_trade_api as tradeapi 

class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        if data_source == "alpaca":
            try:
                API_KEY      = ALPACA_API_KEY
                API_SECRET   = ALPACA_API_SECRET
                API_BASE_URL = ALPACA_API_BASE_URL
                self.processor = Alpaca(
                    API_KEY, API_SECRET, API_BASE_URL)
                ("===============Alpaca successfully connected with class DataProcessor: meta/data_processor.py==========================")
                # @ DOVY DOVprintY
                # API_KEY = kwargs.get("API_KEY")
                # API_SECRET = kwargs.get("API_SECRET")
                # API_BASE_URL = kwargs.get("API_BASE_URL")
                # self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
                # print("Alpaca successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "wrds":
            self.processor = Wrds()

        elif data_source == "yahoofinance":
            self.processor = YahooFinance()

        else:
            raise ValueError("Data source input is NOT supported yet.")

        # Initialize variable in case it is using cache and does not use download_data() method
        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        # print(df)
        # request = StockBarsRequest(
        #     symbol_or_symbols=ticker_list,
        #     timeframe=TimeFrame.Day,
        #     start=start_date,
        #     end=end_date
        # )
        # stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        # df = stock_client.get_stock_bars(request)
        # jprint("Alpaca DOWNLOAD successfully connected with class DataProcessor: meta/data_processor.py++++++++++++++++++++++++++")   
        # jprint(df)     
        return df

    def clean_data(self, df) -> pd.DataFrame:
    
        df = self.processor.clean_data(df)

        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_vixor(self, df) -> pd.DataFrame:
        df = self.processor.add_vixor(df)


        return df

    
    def df_to_array(self, df, tech_indicator_list, if_vix=True):
        """Robust array conversion with validation"""
        # Input validation
        if df.empty:
            raise ValueError("Cannot convert empty DataFrame")
        
        required_cols = ['close', 'tic'] + tech_indicator_list
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Initialize containers
        price_data = []
        tech_data = []
        turbulence_data = []
        
        for tic in sorted(df['tic'].unique()):
            tic_df = df[df['tic'] == tic]
            
            # Validate we have data for this ticker
            if tic_df.empty:
                print(f"Warning: No data for ticker {tic}")
                continue
                
            # Ensure numeric types
            price_data.append(tic_df['close'].values.reshape(-1, 1).astype(np.float32))
            tech_data.append(tic_df[tech_indicator_list].values.astype(np.float32))
            
            if if_vix:
                turbulence_data.append(tic_df['VIXY'].values.astype(np.float32))
            else:
                turbulence_data.append(tic_df['turbulence'].values.astype(np.float32))
        
        # Convert to arrays
        try:
            price_array = np.concatenate(price_data, axis=1)
            tech_array = np.concatenate(tech_data, axis=1)
            turbulence_array = np.concatenate(turbulence_data)
        except Exception as e:
            print(f"Array concatenation failed: {str(e)}")
            raise

        # Final validation
        if price_array.size == 0:
            raise ValueError("Price array is empty after conversion")
            
        return price_array, tech_array, turbulence_array        