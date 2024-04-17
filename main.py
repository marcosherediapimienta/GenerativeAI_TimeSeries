from data_loading_finance.loading_data import LoadingData
import matplotlib.pyplot as plt
from ts_tools.tools import tools
from time_series_gpt.models.Monte_Carlo.prophet import ProphetMeta
from time_series_gpt.models.LLM.Chronos import Chronos
from time_series_gpt.models.Transformers.patchtst import patchtst_forecaster
import pandas as pd
import time

# Load the data
tickers = ['GOOG']
loader = LoadingData(tickers=tickers)
ts = loader.get_data(start_date='2021-01-01', end_date='2022-01-01')
info = loader.get_info_ticker()

ts_tools = tools()
ts = ts_tools.ts_prepartion(ts, 'Date', 'Adj Close')
ts_tools.plot_ts(ts)

chronos = False
patchtst = False
prophet = False

horizon= 30
freq= 'D'

if prophet:
    start_time = time.time()

    model = ProphetMeta(ts_data=ts)
    model.train_and_evaluate()
    ts_forecast = model.predict(horizon=horizon, freq=freq)
    result_metric = model.get_results()

    end_time = time.time()
    prophet_duration = (end_time - start_time)/60

    print(f'Duration of Prophet : {prophet_duration} minutes')

if chronos:
    start_time = time.time()

    model = Chronos(ts_data=ts)
    ts_forecast = model.predict(evaluation=False, horizon=horizon)

    end_time = time.time()
    chronos_duration = (end_time - start_time)/60

    print(f'Duration of Chronos : {chronos_duration} minutes')

if patchtst:
    start_time = time.time()

    model = patchtst_forecaster(evaluation=False, auto=False, input_size=7, freq=freq, horizon=horizon)
    model.fit(ts)
    ts_forecast = model.predict() 

    end_time = time.time()
    patchtst_duration = (end_time - start_time)/60

    print(f'Duration of PatchTST: {patchtst_duration} minutes')   

# Plot the forecast
if not ts_forecast.empty:
    ts_tools.plot_forecast(ts_forecast)