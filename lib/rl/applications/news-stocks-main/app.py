import nltk
nltk.download('vader_lexicon')
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import datetime
import nltk
nltk.download('vader_lexicon')

# Set page title and configure layout
st.set_page_config(page_title="Stock Sentiment Analysis", layout="wide")

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

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

#page title and subtitle
st.title("Stock Sentiment Analysis")
st.markdown("Analyze the sentiment of news headlines and stock price movements for a given stock ticker symbol.")

finviz_url = "https://finviz.com/quote.ashx?t="

#Enter stock ticker symbol
example_ticker_symbols = [
"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
"JPM", "NFLX", "FB", "BRK.B", "V",
"NVDA", "DIS", "BA", "IBM", "GE",
"PG", "JNJ", "KO", "MCD", "T",
"ADBE", "CRM", "INTC", "ORCL", "HD"
]

# Use a selectbox to allow users to choose from example ticker symbols
ticker = st.selectbox("Select a stock ticker symbol or enter your own:", example_ticker_symbols)

news_tables = {}
if ticker:
      #Fetching stock price data
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            stock_data = yf.download(ticker, start="2000-01-01", end=current_date)
      
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
            vader = SentimentIntensityAnalyzer()
            f = lambda title: vader.polarity_scores(title)["compound"]
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
            plt.plot(stock_data.index, stock_data["Close"])
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.title("Stock Price Movements - Line Chart")
            plt.xticks(rotation=45)
            st.subheader("Stock Price Movements - Line Chart")
            st.pyplot(plt)

else:
      st.write("No news found for the entered stock ticker symbol.")

