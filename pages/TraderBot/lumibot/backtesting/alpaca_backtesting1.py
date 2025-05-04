from lumibot.data_sources import AlpacaData
from lumibot.backtesting import BacktestingBroker

class FixedAlpacaBacktesting(AlpacaData):
    """
    Fully operational Alpaca backtesting class that properly integrates with Lumibot
    """
    
    # Mark this as a backtesting broker
    IS_BACKTESTING_BROKER = True
    
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
        # Initialize parent classes
        AlpacaData.__init__(self, config=config)
        BacktestingBroker.__init__(self, data_source=self)
        
        # Store parameters
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        self._backtesting = True
