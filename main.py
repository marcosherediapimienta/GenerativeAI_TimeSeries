from data_loading_finance.loading_data import LoadingData
import matplotlib.pyplot as plt
from ts_tools.tools import tools
from time_series_gpt.models.LLM.Chronos import Chronos
import pandas as pd

# Load the data
loader = LoadingData(tickers=['BTC-USD','GOOG'])
ts = loader.get_data()
info = loader.get_info_ticker()

ts_tools = tools()
ts = ts_tools.ts_prepartion(ts, 'Date', 'Adj Close')
ts_tools.plot_ts(ts)

# Create Chronos model
model = Chronos(ts_data=ts)
ts_forecast = model.predict(evaluation=True)

# Plot the forecast
ts_tools.plot_forecast(ts_forecast)

