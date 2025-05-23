from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import alpaca_trade_api as tradeapi
from alpaca.data.timeframe import TimeFrame

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
from stockstats import StockDataFrame as Sdf
from finrl.agents.stablebaselines3.models import MODELS
from finrl.config import INDICATORS

class FeatureEngineer:
    
    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
        api = None
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.min_data_points = 252  # Minimum data points for robust calculations
    
        from finrl.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL
        if api is None:
            try:
                self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, "v2")
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    def preprocess_data(self, df):
        """Main method to do the feature engineering"""
        if len(df) < self.min_data_points:
            raise ValueError(f"Dataset has {len(df)} rows, but at least {self.min_data_points} are required")

        df = self.clean_data(df)

        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added VIX")
            
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")    

        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user-defined features")

        # Interpolate numeric columns instead of ffill/bfill alone
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='both')
        df = df.ffill().bfill()  # Final fill for edge cases

        # Clip extreme values to prevent infinities
        df[numeric_columns] = df[numeric_columns].clip(lower=-1e10, upper=1e10)

        return df

    def _fetch_data_for_ticker(self, ticker, start_date, end_date, time_interval):
        
        bars = self.api.get_bars(
            ticker,
            time_interval, #: '1D',
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        ).df
        bars["symbol"] = ticker
        return bars

    def download_data(
        self, ticker_list:list, start_date:str, end_date:str, time_interval
    ) -> pd.DataFrame:
        """
        Downloads data using Alpaca's tradeapi.REST method.

        Parameters:
        - ticker_list : list of strings, each string is a ticker
        - start_date : string in the format 'YYYY-MM-DD'
        - end_date : string in the format 'YYYY-MM-DD'
        - time_interval: string representing the interval ('1D', '1Min', etc.)

        Returns:
        - pd.DataFrame with the requested data
        """
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        NY = "America/New_York"
        # start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
        # end_date = pd.Timestamp(end_date     + " 15:59:00", tz=NY)
                
        start_date = pd.Timestamp(start_date).replace(hour=9, minute=30, second=0).tz_localize(NY)
        end_date = pd.Timestamp(end_date).replace(hour=15, minute=59, second=0).tz_localize(NY)

                # Use ThreadPoolExecutor to fetch data for multiple tickers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self._fetch_data_for_ticker,
                    ticker,
                    start_date,
                    end_date,
                    time_interval,
                )
                for ticker in ticker_list
            ]
            data_list = [future.result() for future in futures]

        # Combine the data
        data_df = pd.concat(data_list, axis=0)

        # Convert the timezone
        data_df = data_df.tz_convert(NY)

        # If time_interval is less than a day, filter out the times outside of NYSE trading hours
        if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            data_df = data_df.between_time("09:30", "15:59")

        # Reset the index and rename the columns for consistency
        data_df = data_df.reset_index().rename(
            columns={"index": "timestamp", "symbol": "tic"}
        )

        # Sort the data by both timestamp and tic for consistent ordering
        data_df = data_df.sort_values(by=["tic", "timestamp"])

        # Reset the index and drop the old index column
        data_df = data_df.reset_index(drop=True)

        return data_df

    @staticmethod
    def clean_individual_ticker(args):
        tic, df, times = args
        tmp_df = pd.DataFrame(index=times)
        tic_df = df[df.tic == tic].set_index("timestamp")

        # Step 1: Merging dataframes to avoid loop
        tmp_df = tmp_df.merge(
            tic_df[["open", "high", "low", "close", "volume"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Step 2: Handling NaN values efficiently
        if pd.isna(tmp_df.iloc[0]["close"]):
            first_valid_index = tmp_df["close"].first_valid_index()
            if first_valid_index is not None:
                first_valid_price = tmp_df.loc[first_valid_index, "close"]
                print(
                    f"The price of the first row for ticker {tic} is NaN. It will be filled with the first valid price."
                )
                tmp_df.iloc[0] = [first_valid_price] * 4 + [0.0]  # Set volume to zero
            else:
                print(
                    f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0."
                )
                tmp_df.iloc[0] = [0.0] * 5

        for i in range(1, tmp_df.shape[0]):
            if pd.isna(tmp_df.iloc[i]["close"]):
                previous_close = tmp_df.iloc[i - 1]["close"]
                tmp_df.iloc[i] = [previous_close] * 4 + [0.0]

        # Setting the volume for the market opening timestamp to zero - Not needed
        # tmp_df.loc[tmp_df.index.time == pd.Timestamp("09:30:00").time(), 'volume'] = 0.0

        # Step 3: Data type conversion
        tmp_df = tmp_df.astype(float)

        tmp_df["tic"] = tic

        return tmp_df

    def clean_data(self, df):
        print("Data cleaning started")
        tic_list = np.unique(df.tic.values)
        n_tickers = len(tic_list)

        print("align start and end times")
        grouped = df.groupby("timestamp")
        filter_mask = grouped.transform("count")["tic"] >= n_tickers
        df = df[filter_mask]

        # Generate daily trading days
        trading_days = self.get_trading_days(start=self.start, end=self.end)
        times = [pd.Timestamp(day).tz_localize("America/New_York") for day in trading_days]

        print("Start processing tickers")
        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy(), times))
            future_results.append(result)

        print("ticker list complete")
        print("Start concat and rename")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})
        print("Data clean finished!")
        return new_df

    def add_technical_indicator(
        self,
        df,
        tech_indicator_list=[
            "macd",
            # "boll_ub",
            # "boll_lb",
            # "rsi_30",
            # "dx_30",
            # "close_30_sma",
            # "close_60_sma",
        ],
    ):
        print("Started adding Indicators")

        # Store the original data type of the 'timestamp' column
        original_timestamp_dtype = df["timestamp"].dtype

        # Convert df to stock data format just once
        stock = Sdf.retype(df)
        unique_ticker = stock.tic.unique()

        # Convert timestamp to a consistent datatype (timezone-naive) before entering the loop
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

        print("Running Loop")
        for indicator in tech_indicator_list:
            indicator_dfs = []
            for tic in unique_ticker:
                tic_data = stock[stock.tic == tic]
                indicator_series = tic_data[indicator]

                tic_timestamps = df.loc[df.tic == tic, "timestamp"]

                indicator_df = pd.DataFrame(
                    {
                        "tic": tic,
                        "date": tic_timestamps.values,
                        indicator: indicator_series.values,
                    }
                )
                indicator_dfs.append(indicator_df)

            # Concatenate all intermediate dataframes at once
            indicator_df = pd.concat(indicator_dfs, ignore_index=True)

            # Merge the indicator data frame
            df = df.merge(
                indicator_df[["tic", "date", indicator]],
                left_on=["tic", "timestamp"],
                right_on=["tic", "date"],
                how="left",
            ).drop(columns="date")

        print("Restore Timestamps")
        # Restore the original data type of the 'timestamp' column
        if isinstance(original_timestamp_dtype, pd.DatetimeTZDtype):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["timestamp"] = df["timestamp"].dt.tz_convert(original_timestamp_dtype.tz)
        else:
            df["timestamp"] = df["timestamp"].astype(original_timestamp_dtype)

        print("Finished adding Indicators")
        return df
   
    def add_vix(self, data):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]

        merge_column = "date" if "date" in data.columns else "timestamp"

        vix = vix.rename(
            columns={"timestamp": merge_column, "close": "VIXY"}
        )  # Change column name dynamically

        data = data.copy()
        data = data.merge(
            vix, on=merge_column
        )  # Use the dynamic column name for merging
        data = data.sort_values([merge_column, "tic"]).reset_index(drop=True)

        return data

    def download_and_clean_data(self):
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        return self.clean_data(vix_df)

    def calculate_turbulence(self, data):
        """calculate turbulence index with Alpaca data"""
        df = data.copy()
        
        # 1. Pivot with forward-filled closing prices
        df_price_pivot = df.pivot(index="timestamp", columns='tic', values='close').ffill()
        
        # 2. Calculate returns with NaN handling
        df_returns = df_price_pivot.pct_change().dropna()
        
        # 3. Adjust window size for short datasets
        max_window = min(252, len(df_returns)-1)  # Handle small datasets
        turbulence_index = [0] * (max_window if max_window > 0 else 1)
        
        # 4. Handle insufficient data case
        if len(df_returns) <= 1:
            return pd.DataFrame({"timestamp": df["timestamp"].unique(), 'turbulence': 0})
        
        # 5. Modified turbulence calculation
        for i in range(max_window, len(df_returns)):
            current_date = df_returns.index[i]
            hist_window = df_returns.iloc[i-max_window:i]
            
            # Skip if insufficient data in window
            if len(hist_window) < 5:  # Minimum data points for covariance
                turbulence_index.append(0)
                continue
                
            try:
                cov_matrix = hist_window.cov()
                current_returns = df_returns.iloc[i]
                temp = current_returns.dot(np.linalg.pinv(cov_matrix)).dot(current_returns.T)
                turbulence_index.append(temp.item())
            except:
                turbulence_index.append(0)

        # 6. Align dates with original data
        turbulence_df = pd.DataFrame({
            'timestamp': df_returns.index,
            'turbulence': [0]*max_window + turbulence_index[max_window:]
        })
        
        # 7. Forward-fill missing turbulence values
        full_turbulence = pd.DataFrame({'timestamp': df['timestamp'].unique()})
        turbulence_df = full_turbulence.merge(turbulence_df, on='timestamp', how='left').ffill()
        
        return turbulence_df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_bars([tic], time_interval, limit=limit).df  # [tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_bars(["VIXY"], time_interval, limit=1).df
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb

    def add_turbulence(self, data, time_period=252):
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="timestamp")
        return df.sort_values(["timestamp", "tic"]).reset_index(drop=True)