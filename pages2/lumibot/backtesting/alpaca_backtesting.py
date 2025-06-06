from lumibot.data_sources import AlpacaData, DataSourceBacktesting

class AlpacaBacktesting(DataSourceBacktesting, AlpacaData):
    def __init__(self, datetime_start, datetime_end, **kwargs):
        # raise Exception("AlpacaBacktesting is not currently operational")

        AlpacaData.__init__(self, **kwargs)
        DataSourceBacktesting.__init__(self, datetime_start, datetime_end)
        
# from lumibot.data_sources import AlpacaData


# class AlpacaDataBacktesting(AlpacaData):
#     """
#     AlpacaDataBacktesting is a DataSourceBacktesting that uses AlpacaData as a
#     backtesting data source.
#     """

    def __init__(self, datetime_start, datetime_end, **kwargs):
        AlpacaData.__init__(self, datetime_start, datetime_end, **kwargs)
        
