class tools:
    def __init__(self):
        pass
    def ts_prepartion(self, ts):
        ts['unique_id'] = 'BTC-USD'
        ts = ts.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        ts = ts[['unique_id','ds', 'y']]
        return ts