import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models.bars import BarSet
from alpaca_trade_api import REST
from lib.rl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET


FINVIZ_URL = "https://finviz.com/quote.ashx?t="

# 🔥 Download NLTK dependencies
nltk.download("vader_lexicon")

# 📡 Fetch Real-Time Stock Price
import alpaca_trade_api as tradeapi

def get_real_time_price(ticker):
    try:
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url="https://paper-api.alpaca.markets")
        trade = api.get_latest_trade(ticker)
        return trade.price  # ✅ Return price if trade exists
    except tradeapi.rest.APIError as e:
        print(f"⚠️ Alpaca API Error: {e}")  # ✅ Log the error
        return None  # ✅ Return None if no trade is found

def get_real_time_price9999(ticker):
    api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, base_url="https://data.alpaca.markets/v2")
    trade = api.get_latest_trade(ticker)
    return trade.price

# 🔄 Cached API Client
@st.cache_resource
def get_stock_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# 🔄 Cached API Client
@st.cache_resource
def get_stock_client():
    return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# 📊 Fetch Stock Data from Alpaca
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        client = get_stock_client()
        request_params = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Day, start=start_date, end=end_date)
        return client.get_stock_bars(request_params)
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
        return None

# 📊 Convert Alpaca Data to DataFrame
def convert_alpaca_data_to_df(stock_data):
    """Converts Alpaca BarSet to a Pandas DataFrame for visualization."""
    if isinstance(stock_data, BarSet):
        data_list = []
        for symbol, bars in stock_data.data.items():
            for bar in bars:
                data_list.append({
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume
                })

        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame to avoid errors

# 📰 Fetch News Data from Finviz
@st.cache_data
def fetch_news_data(ticker):
    """Retrieve news headlines from Finviz and ensure date values exist."""
    try:
        req = Request(url=FINVIZ_URL + ticker, headers={"user-agent": "Mozilla/5.0"})
        html = BeautifulSoup(urlopen(req), "html.parser")
        news_table = html.find(id="news-table")

        if not news_table:
            st.warning(f"⚠️ No news data found for {ticker}.")
            return None

        news_list = []
        for row in news_table.findAll("tr"):
            if row.a:
                title = row.a.text.strip()
                date_time_text = row.td.text.strip().split()

                # ✅ Ensure we always have a date
                date = date_time_text[1] if len(date_time_text) > 1 else date_time_text[0] if date_time_text else "Unknown"

                news_list.append({"title": title, "date": date})

        return news_list if news_list else None  # Return None if no valid news found

    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return None



# 🧠 Analyze Sentiment from News
@st.cache_data
def analyze_sentiment(news_list):
    if not news_list:
        return None

    df_sentiment = pd.DataFrame(news_list)
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["date"], errors="coerce").dt.date
    df_sentiment["Compound Score"] = df_sentiment["title"].apply(lambda title: SentimentIntensityAnalyzer().polarity_scores(title)["compound"])
    return df_sentiment


# 📊 Display Sentiment Summary
def display_sentiment_summary(df_sentiment):
    st.subheader("📊 Sentiment Summary")
    avg_score = df_sentiment["Compound Score"].mean() * 100
    sentiment_icon = "😊" if avg_score > 10 else "😢" if avg_score < -10 else "😐"
    st.metric(label=f"💡 Average Sentiment Score {sentiment_icon}", value=f"{avg_score:.2f}%")

    summary = {
        "📈 Positive": f"{(df_sentiment['Compound Score'] > 0).sum() / len(df_sentiment) * 100:.2f}%",
        "📉 Negative": f"{(df_sentiment['Compound Score'] < 0).sum() / len(df_sentiment) * 100:.2f}%",
        "⚖️ Neutral": f"{(df_sentiment['Compound Score'] == 0).sum() / len(df_sentiment) * 100:.2f}%"
    }
    st.json(summary)

# 📈 Plot Stock Data
def plot_stock_data(stock_data, ticker):
    """Processes stock data and plots it."""
    df_stock = convert_alpaca_data_to_df(stock_data)

    if not df_stock.empty:
        st.subheader(f"📈 {ticker} Stock Price Movements")
        st.line_chart(df_stock[["close"]])
    else:
        st.warning("⚠️ No valid stock data available.")
        
# 📈 Compute Moving Averages
def compute_moving_averages(df_stock, close_col):
    """Computes SMA & EMA indicators."""
    df_stock["SMA_50"] = df_stock[close_col].rolling(window=50, min_periods=1).mean()
    df_stock["EMA_20"] = df_stock[close_col].ewm(span=20, adjust=False).mean()
    return df_stock

# 📊 Generate Buy/Sell Signals
def generate_trade_signals(df_stock, df_news, close_col):
    df_news = df_news.copy()

    # ✅ Ensure Date column exists
    if "Date" not in df_news.columns and df_news.index.name == "Date":
        df_news = df_news.reset_index()

    if "Date" not in df_news.columns:
        st.warning("⚠️ 'Date' column missing in df_news. Skipping signal generation.")
        return df_stock

    df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce", utc=True)
    df_news.set_index("Date", inplace=True)

    df_merged = df_stock.merge(df_news[["Compound Score"]], left_index=True, right_index=True, how="left")
    df_merged["Buy_Signal"] = ((df_merged[close_col] > df_merged["SMA_50"]) & (df_merged["Compound Score"] > 0)).astype(int)
    df_merged["Sell_Signal"] = ((df_merged[close_col] < df_merged["SMA_50"]) & (df_merged["Compound Score"] < 0)).astype(int)
    return df_merged
 

# convert_barSet_to_DataFrame
def convert_barSet_to_DataFrame(stock_data, BarSet, ticker):
    df_stock_  = None
    # if isinstance(stock_data, BarSet):
    df_list = []
    for symbol, bars in stock_data.data.items():
        if isinstance(bars, list) and bars:
            df_stock = pd.DataFrame([bar.dict() for bar in bars])
            df_stock["timestamp"] = pd.to_datetime(df_stock["timestamp"])
            df_stock.set_index("timestamp", inplace=True)
            df_stock = df_stock.add_prefix(f"{symbol}_")
            df_list.append(df_stock)

            # DOV debug   
            # st.write("📊 DataFrame Columns:", df_stock.columns)
            if df_list:
                                                
                df_stock_ = pd.concat(df_list, axis=1).loc[:, ~pd.concat(df_list, axis=1).columns.duplicated()]
                
                # 🔍 Debugging: Check column names
                st.write("📊 DataFrame Structure:", df_stock_.head())

                # Ensure we reference the correct column name
                close_col = f"{ticker}_close" if f"{ticker}_close" in df_stock_.columns else None

                if close_col:
                    # ✅ Compute Moving Averages
                    df_stock_["SMA_50"] = df_stock_[close_col].rolling(window=50).mean()
                    df_stock_["EMA_20"] = df_stock_[close_col].ewm(span=20, adjust=False).mean()

                    # ✅ Plot Data
                    st.subheader("📊 Moving Averages & Buy/Sell Signals")
                    st.line_chart(df_stock_[[close_col, "SMA_50", "EMA_20"]])
                else:
                    st.warning(f"⚠️ No closing price data available for {ticker}.")
            else:
                st.warning("⚠️ No valid stock data retrieved! Check API or ticker availability.")
        else:
            st.warning("⚠️ Unexpected data structure received from Alpaca API.")   
            
    return df_stock_ , close_col

def compute_moving_averages(df_stock_, ticker, df_news):
        close_col = f"{ticker}_close" if f"{ticker}_close" in df_stock_.columns else None
        if close_col:
            # ✅ Compute Moving Averages
            df_stock_["SMA_50"] = df_stock_[close_col].rolling(window=50, min_periods=1).mean()
            df_stock_["EMA_20"] = df_stock_[close_col].ewm(span=20, adjust=False).mean()

            # ✅ Convert df_news["Date"] to datetime & set as index
            df_news["Date"] = pd.to_datetime(df_news["Date"], errors="coerce", utc=True)
            df_news.set_index("Date", inplace=True)

            # ✅ Merge Sentiment and Stock Data (Fixed: Ensure both indices are datetime64[ns, UTC])
            df_merged = df_stock_.merge(df_news[["Compound Score"]], left_index=True, right_index=True, how="left")

            # ✅ Generate Buy/Sell Signals
            df_merged["Buy_Signal"] = ((df_merged[close_col] > df_merged["SMA_50"]) & (df_merged["Compound Score"] > 0)).astype(int)
            df_merged["Sell_Signal"] = ((df_merged[close_col] < df_merged["SMA_50"]) & (df_merged["Compound Score"] < 0)).astype(int)

            # ✅ Debugging: Print Buy/Sell Signals to Check if they are valid
            st.write("🔍 Buy/Sell Signal Data Preview:", df_merged[["Buy_Signal", "Sell_Signal"]].tail())

            # ✅ Plot Data (Fixed: Use `df_merged` for plotting)
            st.subheader("📊 Buy/Sell Signals & Moving Averages")
            st.line_chart(df_merged[[close_col, "SMA_50", "EMA_20", "Buy_Signal", "Sell_Signal"]])
        else:
            st.warning(f"⚠️ No closing price data available for {ticker}.")
            
def collapsible_detailed_description(s1,s2):          
  with st.expander(s1):
    st.markdown(s2)  
    
@st.cache_data
def fetch_twitter_sentiment(ticker):
    # Placeholder/mock: In real app, integrate Twitter API or snscrape
    mock_data = [
        {"text": f"{ticker} is going to the moon! 🚀", "sentiment": 0.8},
        {"text": f"I'm not sure about {ticker}, looks weak.", "sentiment": -0.3},
    ]
    df = pd.DataFrame(mock_data)
    df["source"] = "Twitter"
    return df    

@st.cache_data
def fetch_reddit_sentiment(ticker):
    mock_data = [
        {"text": f"{ticker} YOLO play on r/wallstreetbets", "sentiment": 0.7},
        {"text": f"{ticker} is overhyped.", "sentiment": -0.2},
    ]
    df = pd.DataFrame(mock_data)
    df["source"] = "Reddit"
    return df

@st.cache_data
def fetch_google_trends(ticker):
    from pytrends.request import TrendReq
    pytrends = TrendReq()
    kw_list = [ticker]
    pytrends.build_payload(kw_list, timeframe="now 7-d")
    interest = pytrends.interest_over_time()
    if not interest.empty:
        df = interest.reset_index()[["date", ticker]]
        df.rename(columns={ticker: "interest"}, inplace=True)
        return df
    return pd.DataFrame()


    