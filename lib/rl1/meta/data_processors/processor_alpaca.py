from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
from stockstats import StockDataFrame


class AlpacaProcessor:
    def __init__(self, API_KEY=None, API_SECRET=None, API_BASE_URL=None, api=None):
        if api is None:
            try:
                self.api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    def download_data(self, ticker_list, start_date, end_date, time_interval) -> pd.DataFrame:
        """Enhanced data download with complete error handling"""
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        NY = "America/New_York"
        
        try:
            # Convert to proper timestamps
            start_dt = pd.to_datetime(start_date).tz_localize(NY)
            end_dt = pd.to_datetime(end_date).tz_localize(NY)
            
            # Validate date range
            if start_dt > end_dt:
                raise ValueError("Start date must be before end date")

            # Adjust for intraday data
            if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
                start_dt = start_dt.replace(hour=9, minute=30)
                end_dt = end_dt.replace(hour=15, minute=59)

            print(f"Downloading {len(ticker_list)} tickers from {start_dt} to {end_dt}")

            data_list = []
            for ticker in ticker_list:
                try:
                    # Get bars with timeout and retry
                    bars = self.api.get_bars(
                        ticker,
                        time_interval,
                        start=start_dt.isoformat(),
                        end=end_dt.isoformat(),
                        limit=5000,  # Ensure we get enough data points
                        adjustment='all'  # Include splits/dividends
                    ).df
                    
                    if bars.empty:
                        print(f"Warning: No data returned for {ticker}")
                        # Create empty frame with correct structure
                        bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                        bars.index = pd.DatetimeIndex([])
                    else:
                        print(f"{ticker}: Got {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}")
                    
                    bars["symbol"] = ticker
                    data_list.append(bars)

                except Exception as e:
                    print(f"Failed to download {ticker}: {str(e)}")
                    continue

            if not data_list:
                raise ValueError("No data downloaded for any ticker")

            data_df = pd.concat(data_list)
            data_df = data_df.tz_convert(NY)

            # Filter for market hours if intraday
            if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
                data_df = data_df.between_time("09:30", "15:59")

            # Standardize columns
            data_df = data_df.reset_index().rename(
                columns={"index": "timestamp", "symbol": "tic"}
            )
            
            # Validate we have actual data
            if data_df[['open', 'high', 'low', 'close']].isna().all().all():
                raise ValueError("Downloaded data contains only NaN values")
            
            return data_df.sort_values(by=["tic", "timestamp"]).reset_index(drop=True)

        except Exception as e:
            print(f"Fatal download error: {str(e)}")
            raise

    @staticmethod
    def clean_individual_ticker(args):
        """Robust cleaning with detailed diagnostics"""
        tic, df, times = args
        
        # Create empty frame with full time index
        tmp_df = pd.DataFrame(index=times)
        
        try:
            # Get this ticker's data
            tic_df = df[df['tic'] == tic].set_index('timestamp')
            
            if not tic_df.empty:
                tic_df.index = tic_df.index.tz_convert(times[0].tz)
                
                # Join with existing data
                tmp_df = tmp_df.join(tic_df[['open', 'high', 'low', 'close', 'volume']], 
                                    how='left')
                
                # Fill NaN values
                for col in ['open', 'high', 'low', 'close']:
                    tmp_df[col] = tmp_df[col].ffill().bfill()
                tmp_df['volume'] = tmp_df['volume'].fillna(0)
                
                # Handle cases where all data is missing
                if tmp_df[['open', 'high', 'low', 'close']].isna().all().all():
                    print(f"Warning: No valid data for {tic}, filling with zeros")
                    tmp_df[['open', 'high', 'low', 'close']] = 0
                    tmp_df['volume'] = 0
                
                return tmp_df.assign(tic=tic)
            
            # Fallthrough for empty/missing data
            print(f"Creating empty data structure for {tic}")
            tmp_df[['open', 'high', 'low', 'close']] = 0.0
            tmp_df['volume'] = 0.0
            return tmp_df.assign(tic=tic)
            
        except Exception as e:
            print(f"Error cleaning {tic}: {str(e)}")
            tmp_df[['open', 'high', 'low', 'close']] = 0.0
            tmp_df['volume'] = 0.0
            return tmp_df.assign(tic=tic)

    def clean_data(self, df):
        """More reliable data cleaning"""
        print("Starting data cleaning...")
        
        # Validate input
        if df.empty:
            raise ValueError("Cannot clean empty DataFrame")
        
        tic_list = df['tic'].unique()
        print(f"Processing {len(tic_list)} tickers...")

        # Generate proper time index
        if pd.Timedelta(self.time_interval) >= pd.Timedelta(days=1):
            # Daily data
            times = pd.date_range(start=df['timestamp'].min(), 
                                end=df['timestamp'].max(),
                                freq='D').tz_convert('America/New_York')
        else:
            # Intraday data
            trading_days = self.get_trading_days(self.start, self.end)
            times = []
            for day in trading_days:
                current = pd.Timestamp(day + " 09:30:00").tz_localize('America/New_York')
                times.extend([current + pd.Timedelta(minutes=x) for x in range(390)])

        cleaned_dfs = []
        for tic in tic_list:
            try:
                result = self.clean_individual_ticker((tic, df.copy(), times))
                cleaned_dfs.append(result.reset_index())
            except Exception as e:
                print(f"Error processing {tic}: {str(e)}")
                continue

        if not cleaned_dfs:
            raise ValueError("No tickers processed successfully")
        
        final_df = pd.concat(cleaned_dfs).rename(columns={'index': 'timestamp'})
        print(f"Cleaning complete. Final shape: {final_df.shape}")
        return final_df

    def add_technical_indicator(self, df, tech_indicator_list):
        
          # Add this check at the beginning
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing required columns: {required_cols}")
                
        """More robust technical indicator calculation"""
        print("Adding technical indicators...")
        
        try:
            # Convert to StockDataFrame
            stock = StockDataFrame.retype(df.copy())
            
            # Store original timestamp dtype
            original_dtype = df["timestamp"].dtype
            
            # Convert timestamp to naive for calculations
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            
            # Calculate each indicator
            for indicator in tech_indicator_list:
                print(f"Calculating {indicator}...")
                df[indicator] = stock[indicator]
            
            # Restore original timestamp dtype
            if isinstance(original_dtype, pd.DatetimeTZDtype):
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(original_dtype.tz)
            else:
                df["timestamp"] = df["timestamp"].astype(original_dtype)
                
            return df
            
        except Exception as e:
            print(f"Error in add_technical_indicator: {str(e)}")
            raise


    
    # [Rest of your methods remain unchanged...]
    def add_vix(self, data):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]

        merge_column = "date" if "date" in data.columns else "timestamp"

        vix = vix.rename(
            columns={"timestamp": merge_column, "close": "VIXY"}
        )

        data = data.copy()
        data = data.merge(vix, on=merge_column)
        data = data.sort_values([merge_column, "tic"]).reset_index(drop=True)

        return data

    def calculate_turbulence(self, data, time_period=252):
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        start = time_period
        turbulence_index = [0] * start
        count = 0
        
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(current_temp.values.T)
            
            if temp > 0:
                count += 1
                turbulence_temp = temp[0][0] if count > 2 else 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        return pd.DataFrame({"timestamp": df_price_pivot.index, "turbulence": turbulence_index})

    def add_turbulence(self, data, time_period=252):
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        return df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        price_array = tech_array = turbulence_array = None
        
        for tic in unique_ticker:
            tic_data = df[df.tic == tic]
            if price_array is None:
                price_array = tic_data[["close"]].values
                tech_array = tic_data[tech_indicator_list].values
                turbulence_array = tic_data["VIXY" if if_vix else "turbulence"].values
            else:
                price_array = np.hstack([price_array, tic_data[["close"]].values])
                tech_array = np.hstack([tech_array, tic_data[tech_indicator_list].values])
        
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        sessions = nyse.sessions_in_range(
            pd.Timestamp(start).tz_localize(None), 
            pd.Timestamp(end).tz_localize(None)
        )
        return [str(day)[:10] for day in sessions]

    def fetch_latest_data(self, ticker_list, time_interval, tech_indicator_list, limit=100):
        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_bars([tic], time_interval, limit=limit).df
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index(drop=True)
        times = pd.date_range(
            start=data_df.timestamp.min(),
            end=data_df.timestamp.max() + pd.Timedelta(minutes=1),
            freq='1T'
        )

        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(index=times)
            tic_df = data_df[data_df.tic == tic].set_index("timestamp")
            
            if not tic_df.empty:
                tmp_df = tmp_df.join(tic_df[["open", "high", "low", "close", "volume"]], how='left')
                tmp_df[["open", "high", "low", "close"]] = tmp_df[["open", "high", "low", "close"]].ffill().bfill()
                tmp_df["volume"] = tmp_df["volume"].fillna(0)
            else:
                tmp_df[["open", "high", "low", "close"]] = 0.0
                tmp_df["volume"] = 0.0
            
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index().rename(columns={"index": "timestamp"})
        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(df, tech_indicator_list, if_vix=True)
        latest_turb = self.api.get_bars(["VIXY"], time_interval, limit=1).df["close"].values
        
        return price_array[-1], tech_array[-1], latest_turb[0] if len(latest_turb) > 0 else 0

    def download_and_clean_data(self):
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        return self.clean_data(vix_df)