# from lumibot.brokers import Alpaca
# from lumibot.backtesting import YahooDataBacktesting
# from lib.MLTradingBot.lumibot.lumibot.strategies.strategy import Strategy
# from alpaca.trading.client import TradingClient  # Ensure this works
# from lumibot.traders import Trader
from __future__ import annotations
import warnings ; warnings.filterwarnings("ignore")
import streamlit as st
from alpaca_trade_api.rest import REST, TimeFrame
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from lib.MLTradingBot.finbert_utils import estimate_sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentimentIntensityAnalyzerVader = SentimentIntensityAnalyzer()
custom_css = """
<style>
body {
background-color: black; /* Background color (black) */
font-family: "Times New Roman", Times, serif; /* Font family (Times New Roman) */
color: white; /* Text color (white) */
line-height: 1.6; /* Line height for readability */
}

h1 {
color: #3498db; /* Heading color (light blue) */
}

h2 {
color: #e74c3c; /* Subheading color (red) */
}

p {
margin: 10px 0; /* Margin for paragraphs */
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
    "API_KEY": "PKXQGLU5DJJ30MUWS2G6", 
    "API_SECRET": "vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d", 
    "PAPER": True
}
example_ticker_symbols = ["SPY",
"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
"JPM", "NFLX", "FB", "BRK.B", "V",
"NVDA", "DIS", "BA", "IBM", "GE",
"PG", "JNJ", "KO", "MCD", "T",
"ADBE", "CRM", "INTC", "ORCL", "HD"
]
st.title("Sentiment-Based Trading Bot")
st.write("Analyze sentiment and trade stocks using Alpaca API.")
# Use a selectbox to allow users to choose from example ticker symbols
ticker = st.selectbox("Select a stock ticker symbol or enter your own:", example_ticker_symbols)
# User inputs
# symbol = st.text_input("Stock Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
end_date = st.date_input("End Date", value=datetime.now())
sentiment_threshold = st.slider("Sentiment Threshold", -1.0, 1.0, 0.0)
# =====DOV MODE===================
sentiment_threshold = -1 

data = {
    "date": pd.date_range(start=start_date, end=end_date),
    "news": [f"Sample headline for {ticker} on day {i}" for i in range(len(pd.date_range(start=start_date, end=end_date)))],
}
df = pd.DataFrame(data)
df["sentiment"] = df["news"].apply(lambda x: sentimentIntensityAnalyzerVader.polarity_scores(x)["compound"])
st.write("Performing sentiment analysis...")
st.write(df)
st.line_chart(df["sentiment"])
if st.button("Execute Trades"):
    st.write("Executing trades...")
    for index, row in df.iterrows():
        if row["sentiment"] > sentiment_threshold:
            st.write(f"BUY: {ticker} on {row['date']} (Sentiment: {row['sentiment']})")
            # alpaca.submit_order(ticker, qty=1, side="buy", type="market", time_in_force="gtc")
        elif row["sentiment"] < -sentiment_threshold:
            st.write(f"SELL: {ticker} on {row['date']} (Sentiment: {row['sentiment']})")
            # alpaca.submit_order(ticker, qty=1, side="sell", type="market", time_in_force="gtc")

class MLTrader(): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5):
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api =REST(key_id=ALPACA_CREDS["API_KEY"], secret_key=ALPACA_CREDS["API_SECRET"],base_url=BASE_URL)
        st.write(f"Cash at risk: {self.cash_at_risk}")

    def position_sizing(self): 
        # cash = self.get_cash() 
        # last_price = self.get_last_price(self.symbol)
        last_price = 50000 ; cash = 100000; 
        # quantity = round(cash * self.cash_at_risk / last_price,0)
        quantity = 1
        return cash, last_price, quantity

    def get_dates(self): 
        # today = self.get_datetime()
        today = datetime.now()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        st.write(f"Symbol: {self.symbol}   Start: {three_days_prior}  End: {today}")
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        st.write(news)
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()
        if cash > last_price: 
            st.write(f"Probability: {probability}  Sentiment: {sentiment}")
            # if sentiment == "positive" and probability > .999: 
            if probability > .999:     
                if self.last_trade == "sell": 
                    self.api.sell_all() 
                    # self.api.submit_order(
                    #     self.symbol, 
                    #     quantity, 
                    #     side="buy", 
                    #     type="bracket", 
                    #     take_profit=last_price*1.20, 
                    #     stop_loss=last_price*.95
                    #     ) 
                self.last_trade = "buy"
            # elif sentiment == "negative" and probability > .999: 
            elif probability < .999:    
                if self.last_trade == "buy": 
                    self.api.sell_all() 
                    
                # self.api.submit_order(
                # symbol=self.symbol, 
                # qty=quantity, 
                # side="sell", 
                # type="bracket", 
                # take_profit=last_price*.8, 
                # stop_loss=last_price*1.05,                
                # time_in_force="gtc"
                # )
                self.last_trade = "sell"
strat = MLTrader()
strat.initialize(ticker, 0.5)
strat.on_trading_iteration()

news_tables = {}
if ticker:
      #Fetching stock price data
            current_date = datetime.now().strftime("%Y-%m-%d")
            stock_data = yf.download(ticker, start="2000-01-01", end=current_date)
            finviz_url = "https://finviz.com/quote.ashx?t="
            url = finviz_url + ticker
      
            req = Request(url=url, headers={"user-agent": "my-app"})
            response = urlopen(req)
      
            html = BeautifulSoup(response, features="html.parser")
            news_table = html.find(id="news-table")
            news_tables[ticker] = news_table
if news_table:
            parsed_data=[]
            for ticker, news_table in news_tables.items():
              for row in news_table.findAll('tr'):
                  if row.a:
                           title = row.a.text
                           date_data = row.td.text.split()
                           if len(date_data) == 1:
                                  time = date_data[0]
                           else:
                                 date = date_data[1]
                                 time = date_data[0]
                           parsed_data.append([ticker, date, time, title])

      
            df = pd.DataFrame(parsed_data, columns=["Ticker", "Date", "Time", "Headline"])
            f = lambda title: sentimentIntensityAnalyzerVader.polarity_scores(title)["compound"]
            df["Compound Score"] = df["Headline"].apply(f)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
            
            
            # Display data table
            st.subheader("News Headlines and Sentiment Scores")
            st.dataframe(df)
            
            # Sentiment summary
            sentiment_summary = {
            "Average Score": df["Compound Score"].mean(),
            "Positive": (df["Compound Score"] > 0).sum() / len(df) * 100,
            "Negative": (df["Compound Score"] < 0).sum() / len(df) * 100,
            "Neutral": (df["Compound Score"] == 0).sum() / len(df) * 100,
            }
            st.subheader("Sentiment Summary")
            st.write(sentiment_summary)
            
            plt.figure(figsize=(10, 8))
            # dov to numpyh
            # The error indicates that Pandas is no longer supporting multi-dimensional indexing directly on its objects, such as trying to index a DataFrame or Series with [:, None]. Instead, you need to convert the object to a NumPy array before performing such indexing.
            plt.plot(stock_data.index.to_numpy(), stock_data["Close"])
            # plt.plot(stock_data.index[:, None], stock_data["Close"])
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.title("Stock Price Movements - Line Chart")
            plt.xticks(rotation=45)
            st.subheader("Jojo Price Movements - JOJO Chart")
            st.pyplot(plt)

else:
      st.write("No news found for the entered stock ticker symbol.")
