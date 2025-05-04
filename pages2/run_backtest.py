# run_backtest.py
import sys
from datetime import datetime
import streamlit as st
from lumibot.backtesting import YahooDataBacktesting

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 36px !important;
        color: #2E86C1 !important;
        padding-bottom: 15px;
        border-bottom: 2px solid #2E86C1;
    }
    .description {
        font-size: 16px;
        color: #5D6D7E;
        margin-bottom: 25px;
    }
    .results-section {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .metric-box {
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Page configuration
    # st.set_page_config(
    #     page_title="Stock Strategy Backtester",
    #     page_icon="📈",
    #     layout="wide"
    # )

    # Header section
    st.markdown('<h1 class="main-title">Stock Strategy Backtester</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="description">
    This tool analyzes the performance of the Stock Bracket trading strategy against historical market data.
    The backtest compares strategy returns with the S&P 500 (SPY) benchmark for the selected period.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # Strategy information
    with st.expander("ℹ️ About the Strategy"):
        st.markdown("""
        **Stock Bracket Strategy Details**:
        - Entry/exit rules based on technical indicators
        - Automated risk management
        - Fixed position sizing
        - Comparison against benchmark index
        
        _Data Source: Yahoo Finance Historical Prices_
        """)

    # Backtest execution section
    st.subheader("⚙️ Backtest Parameters")
    col1, col2 = st.columns(2)
    with col1:
        backtesting_start = datetime(2023, 1, 1)
        st.write(f"**Start Date:** {backtesting_start.strftime('%Y-%m-%d')}")
    with col2:
        backtesting_end = datetime(2024, 8, 1)
        st.write(f"**End Date:** {backtesting_end.strftime('%Y-%m-%d')}")

    # Run backtest
    from lumibot.example_strategies.stock_bracket import StockBracket
    
    with st.spinner("🚀 Running backtest... This may take a few minutes"):
        results = StockBracket.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            benchmark_asset="SPY",
        )

    # Results display
    st.divider()
    st.subheader("📊 Backtest Results")
    
    if results:
        # Key metrics grid
        cols = st.columns(4)
        with cols[0]:
            st.markdown('<div class="metric-box">📈 **Strategy Return**<br>'
                        f'{results.get("total_return", "N/A"):.2f}%</div>', 
                        unsafe_allow_html=True)
        # with cols[1]:
        #     st.markdown('<div class="metric-box">🏆 **Benchmark Return**<br>'
        #                 f'{results.get("benchmark_return", "N/A"):.2f}%</div>', 
        #                 unsafe_allow_html=True)
        # with cols[2]:
        #     st.markdown('<div class="metric-box">✅ **Winning Trades**<br>'
        #                 f'{results.get("winning_trades", "N/A")}</div>', 
        #                 unsafe_allow_html=True)
        # with cols[3]:
        #     st.markdown('<div class="metric-box">❌ **Losing Trades**<br>'
        #                 f'{results.get("losing_trades", "N/A")}</div>', 
        #                 unsafe_allow_html=True)

        # Detailed results
        with st.expander("🔍 View Detailed Performance Metrics"):
            st.json(results)
            
        st.success("✅ Backtest completed successfully")
    else:
        st.error("❌ No results returned from backtest")

    # Footer
    st.divider()
    st.caption("Disclaimer: Past performance does not guarantee future results. Backtest results are hypothetical.")