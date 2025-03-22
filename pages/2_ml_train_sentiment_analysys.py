# Stock Market Trend Prediction using Sentiment Analysis
# Welcome to the Stock Market Trend Prediction project! This repository contains the code and resources for a cutting-edge approach that combines machine learning algorithms with sentiment analysis to accurately predict stock market trends.
# https://github.com/sardarosama/Stock-Market-Trend-Prediction-Using-Sentiment-Analysis/blob/main/.ipynb_checkpoints/Bert_Training-checkpoint.ipynb
################################################################
# Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis
# ==========
# Stock Market Prediction Web App based on Machine Learning and Sentiment Analysis of Tweets (API keys included in code). The front end of the Web App is based on Flask and Wordpress. The App forecasts stock prices of the next seven days for any given stock under NASDAQ or NSE as input by the user. Predictions are made using three algorithms: ARIMA, LSTM, Linear Regression. The Web App combines the predicted prices of the next seven days with the sentiment analysis of tweets to give recommendation whether the price is going to rise or fall.
# https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/tree/master
######################################################################
# Financial-News-Sentiment-Analysis-Tool
# =======
# https://github.com/IshanDissanayake/Financial-News-Sentiment-Analysis-Tool/blob/main/FinancialNewsSentimentAnalyzer.py
##########################################################################
# Stock-Prediction/scrape/.ipynb_checkpoints
# ========
# /gettingnews-checkpoint.ipynb
# https://github.com/munozalexander/Stock-Prediction/blob/master/scrape/.ipynb_checkpoints/gettingnews-checkpoint.ipynb
##############################
import streamlit as st
import pandas as pd
import numpy as np
# import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from lib.rl.config_tickers import index_dict
from alpaca.data.models.bars import BarSet
from lib.utility.util import (
    get_ticker_start_end_date,
    get_real_time_price,
    fetch_stock_data,
    fetch_news_data,
    analyze_sentiment,
    display_sentiment_summary,
    plot_stock_data,
    compute_moving_averages,
    generate_trade_signals,
    convert_barSet_to_DataFrame,
    compute_moving_averages,
    collapsible_detailed_description,
    load_and_plot_stock_data,
)

import os

st.title("Stock Market Trend Prediction Using Sentiment Analysis")

header = "📖 Analyze stock trends using real-time market data, historical price movements, and AI-powered sentiment analysis."
content =  """   This app combines **machine learning** and **sentiment analysis** to forecast stock market trends using news and price data.

    It integrates:
    - 📊 Stock price data (via Alpaca)
    - 📰 News sentiment analysis
    - 🤖 LSTM model predictions
    - 📈 Technical indicators (SMA, EMA)

    GitHub source references:
    - [Stock Market Trend Prediction project](https://github.com/sardarosama/Stock-Market-Trend-Prediction-Using-Sentiment-Analysis)
    - [Web App with Tweet Sentiment](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
    - [Financial News Sentiment Tool](https://github.com/IshanDissanayake/Financial-News-Sentiment-Analysis-Tool)
    """
# Collapsible Detailed Description
collapsible_detailed_description(header, content)

with st.expander("📘 About this project"):
    st.markdown("""
    This app combines **machine learning** and **sentiment analysis** to forecast stock market trends using news and price data.

    It integrates:
    - 📊 Stock price data (via Alpaca)
    - 📰 News sentiment analysis
    - 🤖 LSTM model predictions
    - 📈 Technical indicators (SMA, EMA)

    GitHub source references:
    - [Stock Market Trend Prediction project](https://github.com/sardarosama/Stock-Market-Trend-Prediction-Using-Sentiment-Analysis)
    - [Web App with Tweet Sentiment](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
    - [Financial News Sentiment Tool](https://github.com/IshanDissanayake/Financial-News-Sentiment-Analysis-Tool)
    """)

ticker, start_date, end_date = get_ticker_start_end_date(index_dict)

# Check if the ticker and start date are provided
if ticker and start_date:
    # Fetch historical stock data
    # 🏆 Main App Logic
    if ticker:
        df_stock_ , close_col = load_and_plot_stock_data(ticker, start_date, end_date)


    fig = px.line(df_stock_, x=df_stock_.index, y=close_col, title=f"{ticker} Stock Price")
    # st.plotly_chart(fig, use_container_width=True, key="data_chart")

    # st.plotly_chart(fig)

    pricing_data, predictions = st.tabs(["Pricing Data", "Predictions"])

    with pricing_data:
        st.header("Price Movement")
        # Fetch stock data
        barset = fetch_stock_data(ticker, start_date, end_date)
        data2, close_col = load_and_plot_stock_data(ticker, start_date, end_date)
        # Ensure DataFrame is valid before proceeding
        if data2 is None or data2.empty or close_col not in data2.columns:
            st.error("⚠️ No usable data available for this stock.")
            st.stop()

        # Compute percentage change
        data2["% Change"] = data2[close_col] / data2[close_col].shift(1) - 1

        # Drop NaN values
        data2.dropna(inplace=True)
        st.write(data2)
        annual_return = data2['% Change'].mean() * 252 * 100 - 1
        st.write('Annual Return:', annual_return, '%')
        stdev = np.std(data2['% Change']) * np.sqrt(252)
        st.write('Standard Deviation:', stdev * 100, '%')
        st.write('Risk-Adjusted Return:', annual_return / (stdev * 100))

        with predictions:
            st.header("Predictions")
            # Display the fetched data in a line chart
            st.plotly_chart(fig, use_container_width=True, key="predictions")

else:
    st.write("Please enter a valid ticker and start date.")

# App 2: Stock Trend Prediction

st.title("Stock Trend Prediction")

start = "2011-02-01"
end = "2019-12-31"
user_input = st.text_input("Enter Stock Ticker", "AAPL")
# df = yf.download(user_input, start=start, end=end)
df = fetch_stock_data(ticker, start_date, end_date)

barset = fetch_stock_data(ticker, start_date, end_date)
df, close_col = convert_barSet_to_DataFrame(barset, None, ticker)

if df is None or df.empty:
    st.error("⚠️ No stock data to display.")
    st.stop()

# ✅ Now it's safe to reset the index if needed
# df = df.reset_index()
df = df.reset_index()
df = df.dropna()

# Describing data
st.subheader('Data from 2011-2019')
st.write("Description")
st.write(df.describe())

# Visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
# plt.plot(df.Close)
plt.plot(df[close_col])
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA")
# ma100 = df.Close.rolling(100).mean()
ma100 = df[close_col].rolling(100).mean()
if close_col not in df.columns:
    st.error(f"⚠️ Column '{close_col}' not found in data.")
    st.stop()

fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
# plt.plot(df.Close)
plt.plot(df[close_col])
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100 = df[close_col].rolling(100).mean()
ma200 = df[close_col].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "b")
plt.plot(ma200, "g")
plt.plot(df[close_col])
st.pyplot(fig)

split = int(len(df) * 0.70)

data_training = pd.DataFrame(df[close_col][:split])
data_testing = pd.DataFrame(df[close_col][split:])

if len(data_testing) > 0:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    # // jojostock1/pages/Stock-Prediction/models/model0.h5
    # jojostock1/pages/Stock-Prediction/models/model0.h5
    model_path = "pages/Stock-Prediction/models/model0.h5"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    else:
        model = load_model(model_path)  # or torch.load(model_path) depending on your framework    
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the file 'stock_sentiment_model.pt.h5' exists.")
    st.stop()  # Stop further execution

    # Feeding model with past 100 days of data
    # Testing part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_pred = model.predict(x_test)

    scale_factor = 1 / 0.13513514
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader("Prediction vs Original")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_pred, 'r', label='Predicted Price')
    plt.plot(y_test, 'b', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
else:
    st.write("Insufficient data for testing.")