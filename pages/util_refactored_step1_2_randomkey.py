
import streamlit as st
import plotly.express as px
from lib.utility.util import fetch_stock_data, convert_barSet_to_DataFrame
import random

def load_and_plot_stock_data(ticker, start_date, end_date, plot=True):
    barset = fetch_stock_data(ticker, start_date, end_date)

    if barset is None:
        st.error("⚠️ Failed to fetch stock data.")
        return None, None

    df, close_col = convert_barSet_to_DataFrame(barset, None, ticker)

    if df is None or df.empty or close_col not in df.columns:
        st.warning(f"⚠️ No valid data for {ticker}.")
        return None, None

    if plot:
        chart_key = f"stock_price_chart_{random.randint(1000, 9999)}"
        fig = px.line(df, x=df.index, y=close_col, title=f"{ticker} Stock Price")
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    return df, close_col

def compute_price_features(df, close_col, short_window=50, long_window=100):
    if close_col not in df.columns:
        st.error(f"⚠️ '{close_col}' not found in data.")
        return df

    df["SMA_50"] = df[close_col].rolling(window=short_window, min_periods=1).mean()
    df["EMA_20"] = df[close_col].ewm(span=20, adjust=False).mean()
    df["SMA_100"] = df[close_col].rolling(window=long_window, min_periods=1).mean()
    df["% Change"] = df[close_col].pct_change()

    df.dropna(inplace=True)
    return df
