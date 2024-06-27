import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from tsfeatures import tsfeatures
from tsfeatures import acf_features

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose 


from ts_tools.tools import tools
from data_loading_finance.loading_data import LoadingData

df = pd.read_csv("/Users/marcosherediapimienta/Library/Mobile Documents/com~apple~CloudDocs/Documents/Máster de Matemàtiques per els Instruments Financers/TFM/Time_Series/archive/Top10-2021-2024-1d.csv")
df.head()

# Seleccionar las columnas deseadas del DataFrame original
df_selected = df[['Timestamp', 'BTCUSDT']]
df_selected = df_selected.rename(columns={'Timestamp':'date', 'BTCUSDT': 'BTC'})
df_selected = df_selected.dropna()
print(df_selected)

df_selected["unique_id"]="1"
df_selected.columns=["ds", "y", "unique_id"]
df_selected.head()

print(df_selected.dtypes)

df_selected["ds"] = pd.to_datetime(df_selected["ds"])
print(df_selected.dtypes)

StatsForecast.plot(df_selected, engine="matplotlib")

# Crear una figura con dos subgráficos
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Graficar ACF en el primer subgráfico
plot_acf(df_selected["y"], lags=60, ax=axs[0], color="blue")
axs[0].set_title("Autocorrelation")

# Graficar PACF en el segundo subgráfico
plot_pacf(df_selected["y"], lags=60, ax=axs[1], color="blue")
axs[1].set_title('Partial Autocorrelation')

# Ajustar el diseño para que no haya superposición de elementos
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Descomponer la serie temporal
decomposition = seasonal_decompose(df_selected['y'], model='additive', period=12)
decomposition.plot()
plt.show()

Y_train_df = df_selected[df_selected.ds<='2024-05-01'] 
Y_test_df = df_selected[df_selected.ds>'2024-05-01']
Y_train_df.shape, Y_test_df.shape

# Ajustar el tamaño de la figura
plt.figure(figsize=(12, 6))

# Graficar las series temporales de entrenamiento y prueba
sns.lineplot(data=Y_train_df, x="ds", y="y", label="Train")
sns.lineplot(data=Y_test_df, x="ds", y="y", label="Test")

# Personalizar etiquetas y título
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Train vs Test Data")
plt.legend()
plt.tight_layout()  
plt.show()

season_length = 12 # Monthly data 
horizon = len(Y_test_df) # number of predictions

models = [AutoARIMA(season_length=season_length)]

sf = StatsForecast(df=Y_train_df,
                   models=models,
                   freq='D', 
                   n_jobs=-1)

sf.fit()

arima_string(sf.fitted_[0,0].model_)

result=sf.fitted_[0,0].model_
print(result.keys())
print(result['arma'])

residual=pd.DataFrame(result.get("residuals"), columns=["residual Model"])
residual

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plot de Residuos
residual['residual Model'].plot(ax=axs[0, 0])
axs[0, 0].set_title("Residuals")
axs[0, 0].set_xlabel('Time')

# Density plot - Residuos
sns.distplot(residual['residual Model'], ax=axs[0, 1])
axs[0, 1].set_title("Density plot - Residuals")

# Q-Q Plot de Residuos
stats.probplot(residual['residual Model'], dist="norm", plot=axs[1, 0])
axs[1, 0].set_title('Q-Q Plot')

# Autocorrelation de Residuos
plot_acf(residual['residual Model'], lags=35, ax=axs[1, 1], color="blue")
axs[1, 1].set_title("Autocorrelation of Residuals")

plt.tight_layout()
plt.show()

forecast_df = sf.forecast(h=28, level = [95]) 
forecast_df

df_plot=pd.concat([df_selected, forecast_df]).set_index('ds').tail(75)
df_plot

# Asumiendo que df_plot contiene las columnas necesarias y está preparado para trazar

fig, ax = plt.subplots(1, 1, figsize=(20, 8))

# Graficar las series y la predicción
plt.plot(df_plot.index, df_plot['y'], 'k--', label="Actual", linewidth=2)
plt.plot(df_plot.index, df_plot['AutoARIMA'], 'b-', label="AutoARIMA Forecast", linewidth=2, color="red")

ax.fill_between(df_plot.index, 
                df_plot['AutoARIMA-lo-95'], 
                df_plot['AutoARIMA-hi-95'],
                alpha=.2,
                color='red',
                label='AutoARIMA 95% Confidence Interval')

# Configurar título y etiquetas de los ejes
ax.set_title('Daily Bitcoin Price Forecast', fontsize=20)
ax.set_ylabel('BTC Price', fontsize=15)
ax.set_xlabel('Date', fontsize=15)

# Añadir leyenda
ax.legend(prop={'size': 12})

# Mostrar la cuadrícula
ax.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()


