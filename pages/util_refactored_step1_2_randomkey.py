
import streamlit as st
import plotly.express as px
from lib.utility.util import fetch_stock_data, convert_barSet_to_DataFrame
import random

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
