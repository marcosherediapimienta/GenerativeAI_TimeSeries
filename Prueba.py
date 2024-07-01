import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, AutoCES, AutoTheta
from time import time
import seaborn as sns
from scipy import stats
import warnings

df = pd.read_csv("/Users/marcosherediapimienta/Library/Mobile Documents/com~apple~CloudDocs/Documents/Máster de Matemàtiques per els Instruments Financers/TFM/Time_Series/archive/Top10-2021-2024-1d.csv")

df_selected = df[['Timestamp', 'BTCUSDT']]
df_selected = df_selected.rename(columns={'Timestamp':'ds', 'BTCUSDT': 'y'})
df_selected = df_selected.dropna()

df_selected["unique_id"] = "1"
df_selected.columns = ["ds", "y", "unique_id"]

df_selected["ds"] = pd.to_datetime(df_selected["ds"])
df_selected

Y_train_df = df_selected[df_selected.ds <= '2024-05-01']
Y_test_df = df_selected[df_selected.ds > '2024-05-01']

season_length = 12
horizon = len(Y_test_df)
models = [AutoETS(season_length=season_length), AutoARIMA(season_length=season_length), AutoCES(season_length=season_length), AutoTheta(season_length=season_length)]

sf = StatsForecast(df=Y_train_df,
                   models=models,
                   freq='D',
                   n_jobs=-1)

sf.fit()

horizon = 28
levels = [99] 

forecast_df = sf.forecast(h=horizon, level = levels, fitted = True)
forecast_df = forecast_df.reset_index()
forecast_df.head()

# Asegurarse de que la columna 'ds' está presente en ambos DataFrames
if 'ds' not in Y_test_df.columns:
    Y_test_df = Y_test_df.reset_index().rename(columns={'index': 'ds'})
if 'ds' not in forecast_df.columns:
    forecast_df = forecast_df.reset_index().rename(columns={'index': 'ds'})

# Configurar los índices para los cálculos de error
Y_test_df = Y_test_df.set_index('ds')
forecast_df = forecast_df.set_index('ds')

# Asegurarse de que las fechas en los índices coincidan
Y_test_df = Y_test_df.loc[forecast_df.index]

# Definir funciones para calcular MAPE y sMAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# Inicializar diccionarios para almacenar los errores de cada modelo
mape_results = {}
smape_results = {}

# Lista de nombres de los modelos
model_names = ['AutoETS','AutoARIMA', 'CES', 'AutoTheta']

# Calcular MAPE y sMAPE para cada modelo
for model_name in model_names:
    y_pred = forecast_df[model_name].values
    y_true = Y_test_df['y'].values
    mape_results[model_name] = mean_absolute_percentage_error(y_true, y_pred)
    smape_results[model_name] = symmetric_mean_absolute_percentage_error(y_true, y_pred)

# Mostrar los resultados
print("MAPE results:")
for model, mape in mape_results.items():
    print(f"{model}: {mape:.2f}%")

print("\nSMAPE results:")
for model, smape in smape_results.items():
    print(f"{model}: {smape:.2f}%")
