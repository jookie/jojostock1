from lumibot.data_sources import AlpacaData
from lumibot.backtesting import BacktestingBroker

IS_BACKTESTING_BROKER = True


class WorkingAlpacaBacktesting(AlpacaData):
    """
    Operational replacement for the non-working AlpacaBacktesting class
    """
    
    def __init__(self, datetime_start, datetime_end, **kwargs):
        # Initialize config
        config = kwargs.pop('config', {}) or {}
        config.update({
            'datetime_start': datetime_start,
            'datetime_end': datetime_end,
            'is_backtesting': True,
            'API_KEY': kwargs.pop('API_KEY', 'backtest_dummy'),
            'API_SECRET': kwargs.pop('API_SECRET', 'backtest_dummy'),
            'PAPER': kwargs.pop('PAPER', True),
        })
        
        # Initialize parent class
        super().__init__(config=config)
        
        # Store backtesting parameters
        self._datetime_start = datetime_start
        self._datetime_end = datetime_end
        self._backtesting = True