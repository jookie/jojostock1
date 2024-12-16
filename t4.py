
# Set page title and configure layout
# st.set_page_config(page_title="Stock Sentiment Analysis", layout="wide")
# #page title and subtitle
# st.title("Stock Sentiment Analysis - Jojo")
# st.markdown("Analyze the sentiment of news headlines and stock price movements for a given stock ticker symbol.")
finviz_url = "https://finviz.com/quote.ashx?t="
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
            current_date = datetime.now().strftime("%Y-%m-%d")
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
            #DOV to numpy
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


# User inputs
symbol = st.text_input("Stock Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
end_date = st.date_input("End Date", value=datetime.now())
sentiment_threshold = st.slider("Sentiment Threshold", -1.0, 1.0, 0.0)