import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline
from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.traders import Trader

# Initialize BERT sentiment analyzer
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

sentiment_analyzer = load_sentiment_model()

# Custom Lumibot Strategy
class SentimentTradingBot(Strategy):
    def initialize(self, params):
        self.params = params
        self.sleeptime = "1D"
        self.set_asset(self.params['symbol'])
        
    def on_trading_iteration(self):
        try:
            # Get sentiment from chat history
            latest_news = self.params['news'][-1] if self.params['news'] else ""
            sentiment = sentiment_analyzer(latest_news)[0]
            
            # Get technical indicators
            rsi = self.get_rsi()
            
            # Trading logic
            portfolio_value = self.get_portfolio_value()
            position = self.get_position(self.asset)
            
            if sentiment['label'] == 'POSITIVE' and rsi < 30:
                if not position:
                    order = self.create_order(self.asset, quantity=portfolio_value//self.get_last_price(self.asset))
                    self.submit_order(order)
            elif sentiment['label'] == 'NEGATIVE' and rsi > 70:
                if position:
                    self.sell_all()
                    
        except Exception as e:
            self.log_message(f"Error: {str(e)}")

# Chat interface
st.title("🤖 Financial AI Assistant")
st.caption("Ask about market sentiment or type '/backtest' to run strategies")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process commands
    if prompt.startswith("/backtest"):
        try:
            # Parse command
            _, symbol, start_date, end_date = prompt.split()
            
            # Show loading state
            with st.chat_message("assistant"):
                st.markdown(f"🚀 Running backtest for {symbol} from {start_date} to {end_date}...")
                with st.spinner("Analyzing market data..."):
                    # Configure backtest
                    backtesting = YahooDataBacktesting(
                        start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                        end_date=datetime.strptime(end_date, "%Y-%m-%d"),
                    )

                    # Run strategy
                    bot = SentimentTradingBot(
                        backtesting=backtesting,
                        budget=100000,
                        params={
                            'symbol': symbol,
                            'news': [m['content'] for m in st.session_state.messages if m['role'] == 'user']
                        }
                    )
                    
                    results = bot.backtest()
                    
                    # Display results
                    st.success("Backtest completed!")
                    
                    # Performance metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{results['total_return']:.2%}")
                    col2.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    
                    # Plot returns
                    fig, ax = plt.subplots()
                    results['returns'].plot(ax=ax)
                    ax.set_title("Portfolio Returns")
                    st.pyplot(fig)
                    
                    response = f"Backtest results for {symbol} ready!"
            
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    else:
        # Analyze sentiment
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    result = sentiment_analyzer(prompt)[0]
                    emoji = "📈" if result['label'] == 'POSITIVE' else "📉"
                    response = f"{emoji} Sentiment: {result['label']} (Confidence: {result['score']:.2%})"
                except:
                    response = "🤖 Hello! How can I assist with market analysis today?"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)