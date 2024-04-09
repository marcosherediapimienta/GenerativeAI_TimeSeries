from data_loading_finance.loading_data import LoadingData
import matplotlib.pyplot as plt
from ts_tools.tools import tools

# Load the data
loader = LoadingData(tickers=['BTC-USD','GOOG'])
ts = loader.get_data()
info = loader.get_info_ticker()

ts_tools = tools()
ts = ts_tools.ts_prepartion(ts, 'Date', 'Adj Close')
ts_tools.plot(ts)
