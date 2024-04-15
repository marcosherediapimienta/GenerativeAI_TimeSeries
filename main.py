from data_loading_finance.loading_data import LoadingData
import matplotlib.pyplot as plt
from ts_tools.tools import tools
from time_series_gpt.models.Monte_Carlo.prophet import ProphetMeta
from time_series_gpt.models.LLM.Chronos import Chronos
from time_series_gpt.models.Transformers.patchtst import patchtst_forecaster
import pandas as pd

# Load the data
loader = LoadingData(tickers=['BTC-USD','GOOG'])
ts = loader.get_data()
info = loader.get_info_ticker()

ts_tools = tools()
ts = ts_tools.ts_prepartion(ts, 'Date', 'Adj Close')
# ts_tools.plot_ts(ts)

chronos = False
patchtst = False
prophet = True

horizon= 30
freq= 'D'

if prophet:
    model = ProphetMeta(ts_data=ts)
    model.train_and_evaluate()
    ts_forecast = model.predict(horizon=horizon, freq=freq)
    result_metric = model.get_results()

if chronos:
    model = Chronos(ts_data=ts)
    ts_forecast = model.predict(evaluation=True)

if patchtst:
    model = patchtst_forecaster()
    ts_fitting = model.fit(ts,evaluation=True)
    ts_forecast = model.predict()    

# Plot the forecast
if not ts_forecast.empty:
    ts_tools.plot_forecast(ts_forecast)

