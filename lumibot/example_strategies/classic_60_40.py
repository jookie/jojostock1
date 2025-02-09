from datetime import datetime

from lumibot.credentials import IS_BACKTESTING
from lumibot.example_strategies.drift_rebalancer import DriftRebalancer

"""
Strategy Description:

This script implements a "Drift Rebalancer" strategy for a classic 60/40 portfolio.
It maintains a portfolio consisting of:
    - 60% stocks (SPY - S&P 500 ETF)
    - 40% bonds (TLT - Long-term US Treasury ETF)

The strategy monitors the portfolio daily and rebalances when the asset allocation 
devitates beyond a specified threshold (5%).
If the weight of an asset drifts beyond this threshold, the strategy sells the overweight asset 
and buys the underweight asset to restore the target balance.

This script is designed for **backtesting only** and will not run in live trading mode.
"""

if __name__ == "__main__":
    # Ensure that the script is running in backtesting mode
    if not IS_BACKTESTING:
        print("This strategy is not meant to be run live. Please set IS_BACKTESTING to True.")
        exit()
    else:
        # Define strategy parameters
        parameters = {
            "market": "NYSE",  # Market where the strategy is executed
            "sleeptime": "1D",  # Time interval for rebalancing (daily)
            "drift_threshold": "0.05",  # Rebalance when assets drift beyond 5% of target weights
            "acceptable_slippage": "0.005",  # 0.5% slippage tolerance for order execution
            "fill_sleeptime": 15,  # Order fill wait time in seconds
            "target_weights": {
                "SPY": "0.60",  # 60% allocated to SPY (stocks)
                "TLT": "0.40"   # 40% allocated to TLT (bonds)
            },
            "shorting": False  # Short selling is not allowed
        }

        from lumibot.backtesting import YahooDataBacktesting

        # Define backtesting start and end dates
        backtesting_start = datetime(2023, 1, 2)  # Backtesting starts from Jan 2, 2023
        backtesting_end = datetime(2024, 10, 31)  # Backtesting runs until Oct 31, 2024

        # Run the backtest using Yahoo Finance data
        results = DriftRebalancer.backtest(
            YahooDataBacktesting,  # Data source for historical prices
            backtesting_start,  # Start date for backtest
            backtesting_end,  # End date for backtest
            benchmark_asset="SPY",  # Compare performance against SPY (S&P 500 ETF)
            parameters=parameters,  # Pass strategy parameters
            show_plot=False,  # Disable plot visualization
            show_tearsheet=False,  # Disable performance report generation
            save_tearsheet=False,  # Do not save performance report
            show_indicators=False,  # Do not display additional indicators
            save_logfile=False,  # Do not save logs
            # show_progress_bar=False,  # Uncomment to hide progress bar
            # quiet_logs=False  # Uncomment to show detailed logs
        )

        # Print backtest results
        print(results)
