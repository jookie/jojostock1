import streamlit as st
from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta
from lib.MLTradingBot.finbert_utils import estimate_sentiment
import pandas as pd

# Alpaca API credentials and initialization
BASE_URL = "https://paper-api.alpaca.markets"  # Use Alpaca paper trading for testing
ALPACA_CREDS = {
    "API_KEY": "PKXQGLU5DJJ30MUWS2G6",
    "API_SECRET": "vPSm9TeqjD7WhYYcuhhvdyXZiFjJQDSlO5ic5s1d",
    "PAPER": True
}

# List of DJI 30 stocks
DJI_30_SYMBOLS = [
    "AAPL", "MSFT", "JPM", "JNJ", "V", "UNH", "PG", "HD", "DIS", "INTC",
    "WMT", "IBM", "MMM", "KO", "MRK", "NKE", "PFE", "TRV", "CSCO", "AXP",
    "VZ", "BA", "CVX", "XOM", "CAT", "MCD", "GS", "WBA", "DOW", "AMGN", "SPY"
]

# Streamlit app title
st.title("Sentiment-Based Trading Bot with Backtest and Live Trading")

# Initialize Alpaca REST API
api = REST(
    key_id=ALPACA_CREDS["API_KEY"],
    secret_key=ALPACA_CREDS["API_SECRET"],
    base_url=BASE_URL
)

# Default dates for date inputs
today = datetime.now()
three_days_ago = today - timedelta(days=3)

# Create a row with symbol selection and date inputs
st.subheader("News and Date Selection")
col1, col2, col3 = st.columns([1, 1, 1])

# Symbol selection in the first column
with col1:
    symbol = st.selectbox("Select Symbol for News", DJI_30_SYMBOLS, index=DJI_30_SYMBOLS.index("SPY"))

# Start date input in the second column
with col2:
    start_date = st.date_input("Start Date", value=three_days_ago)

# End date input in the third column
with col3:
    end_date = st.date_input("End Date", value=today)

# Validate date range
if start_date >= end_date:
    st.error("Start Date must be before End Date!")
else:
    # Fetch news and process data
    try:
        news1 = api.get_news(symbol=symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        parsed_data = []

        # Process each news article
        for article in news1:
            sentiment_scores = estimate_sentiment(article.headline)
            positive_score = float(sentiment_scores[0])  # Convert tensor to float
            parsed_data.append({
                "Ticker": ", ".join(article.symbols),
                "Date": article.created_at.strftime('%Y-%m-%d'),
                "Time": article.created_at.strftime('%H:%M:%S'),
                "Headline": article.headline,
                "Sentiment Score": positive_score
            })
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        parsed_data = []

    # Compute sentiment summary and make trading decisions
    if parsed_data:
        df = pd.DataFrame(parsed_data)
        avg_score = df["Sentiment Score"].mean()
        positive_count = (df["Sentiment Score"] > 0).sum()
        negative_count = (df["Sentiment Score"] < 0).sum()
        neutral_count = (df["Sentiment Score"] == 0).sum()

        sentiment_summary = {
            "Average Score": avg_score,
            "Positive": positive_count,
            "Negative": negative_count,
            "Neutral": neutral_count
        }

        # Display sentiment summary
        st.subheader("Sentiment Summary")
        st.json(sentiment_summary)

        # Display news in a table
        st.dataframe(df)

        # Backtesting feature
        st.subheader("Backtesting")
        backtest_start_date = st.date_input("Backtest Start Date", value=three_days_ago)
        backtest_end_date = st.date_input("Backtest End Date", value=today)

        if backtest_start_date >= backtest_end_date:
            st.error("Backtest Start Date must be before Backtest End Date!")
        else:
            try:
                backtest_news = api.get_news(symbol=symbol, start=backtest_start_date.strftime('%Y-%m-%d'), end=backtest_end_date.strftime('%Y-%m-%d'))
                backtest_data = []

                for article in backtest_news:
                    sentiment_scores = estimate_sentiment(article.headline)
                    positive_score = float(sentiment_scores[0])  # Convert tensor to float
                    # Use Alpaca API to fetch real price data for each news date
                    price_data = api.get_bars(symbol, "day", start=article.created_at.strftime('%Y-%m-%d'), end=article.created_at.strftime('%Y-%m-%d')).df
                    close_price = price_data['close'].iloc[0] if not price_data.empty else None
                    backtest_data.append({
                        "Date": article.created_at.strftime('%Y-%m-%d'),
                        "Time": article.created_at.strftime('%H:%M:%S'),
                        "Sentiment Score": positive_score,
                        "Price": close_price,
                        "Headline": article.headline
                    })

                backtest_df = pd.DataFrame(backtest_data).dropna()

                # Simulate backtesting
                starting_balance = st.number_input("Starting Balance ($)", value=10000.0)
                balance = starting_balance
                holdings = 0
                trade_log = []

                for _, row in backtest_df.iterrows():
                    if row["Sentiment Score"] > 0.5:
                        # Simulate buy
                        trade_log.append({"Action": "BUY", "Date": row["Date"], "Time": row["Time"], "Price": row["Price"]})
                        balance -= row["Price"]
                        holdings += 1
                    elif row["Sentiment Score"] < -0.5 and holdings > 0:
                        # Simulate sell
                        trade_log.append({"Action": "SELL", "Date": row["Date"], "Time": row["Time"], "Price": row["Price"]})
                        balance += row["Price"]
                        holdings -= 1

                # Final balance calculation
                final_balance = balance + holdings * backtest_df["Price"].iloc[-1] if holdings > 0 else balance

                st.write(f"Starting Balance: ${starting_balance:.2f}")
                st.write(f"Final Balance: ${final_balance:.2f}")
                st.write(f"Net Profit/Loss: ${final_balance - starting_balance:.2f}")

                # Display trade log
                st.subheader("Trade Log")
                st.dataframe(pd.DataFrame(trade_log))

                # Generate tear sheet
                st.subheader("Tear Sheet")
                tear_sheet_data = {
                    "Total Trades": len(trade_log),
                    "Winning Trades": sum(1 for trade in trade_log if trade["Action"] == "SELL" and trade["Price"] > starting_balance / len(trade_log)),
                    "Losing Trades": sum(1 for trade in trade_log if trade["Action"] == "SELL" and trade["Price"] < starting_balance / len(trade_log)),
                    "Final Balance": final_balance,
                    "Net Profit/Loss": final_balance - starting_balance
                }
                st.json(tear_sheet_data)

            except Exception as e:
                st.error(f"Error during backtesting: {e}")

        # Live trading
        st.subheader("Live Trading")
        sentiment_threshold = st.slider(
            "Set Sentiment Threshold for Trading",
            min_value=-1.0,
            max_value=1.0,
            value=0.5
        )

        if avg_score > sentiment_threshold:
            st.write(f"Average sentiment score is {avg_score:.2f}, which is above the threshold. Executing BUY order...")
            try:
                # Place a market buy order for the selected symbol
                api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side="buy",
                    type="market",
                    time_in_force="gtc"  # Good 'til canceled
                )
                st.success("Buy order placed successfully!")
            except Exception as e:
                st.error(f"Error placing buy order: {e}")
        elif avg_score < -sentiment_threshold:
            st.write(f"Average sentiment score is {avg_score:.2f}, which is below the threshold. Executing SELL order...")
            try:
                # Place a market sell order for the selected symbol
                api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side="sell",
                    type="market",
                    time_in_force="gtc"  # Good 'til canceled
                )
                st.success("Sell order placed successfully!")
            except Exception as e:
                st.error(f"Error placing sell order: {e}")
        else:
            st.write(f"Average sentiment score is {avg_score:.2f}, which does not meet the threshold for trading.")
    else:
        st.write("No news data available for the given date range.")
