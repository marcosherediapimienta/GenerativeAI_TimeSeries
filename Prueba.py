import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import seaborn as sns
from scipy import stats
import warnings

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Emitir una advertencia personalizada
warnings.warn("Este es un mensaje de advertencia.")

# Carga y preparación de datos
df = pd.read_csv("/Users/marcosherediapimienta/Library/Mobile Documents/com~apple~CloudDocs/Documents/Máster de Matemàtiques per els Instruments Financers/TFM/Time_Series/archive/Top10-2021-2024-1d.csv")

df_selected = df[['Timestamp', 'BTCUSDT']]
df_selected = df_selected.rename(columns={'Timestamp':'ds', 'BTCUSDT': 'y'})
df_selected = df_selected.dropna()

df_selected["unique_id"] = "1"
df_selected.columns = ["ds", "y", "unique_id"]

df_selected["ds"] = pd.to_datetime(df_selected["ds"])

# Análisis de autocorrelación y descomposición estacional
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
plot_acf(df_selected["y"], lags=60, ax=axs[0], color="blue")
axs[0].set_title("Autocorrelación")
plot_pacf(df_selected["y"], lags=60, ax=axs[1], color="blue")
axs[1].set_title('Autocorrelación parcial')

decomposition = seasonal_decompose(df_selected['y'], model='additive', period=12)
decomposition.plot()
plt.show()

# Separación de los datos en conjuntos de entrenamiento y prueba
Y_train_df = df_selected[df_selected.ds <= '2024-05-01']
Y_test_df = df_selected[df_selected.ds > '2024-05-01']

# Configuración y ajuste del modelo AutoARIMA
season_length = 12
horizon = len(Y_test_df)
models = [AutoARIMA(season_length=season_length)]

sf = StatsForecast(df=Y_train_df,
                   models=models,
                   freq='D',
                   n_jobs=-1)

sf.fit()

# Análisis del modelo ajustado y sus residuos
result = sf.fitted_[0,0].model_
print(result.keys())
print(result['arma'])

residual = pd.DataFrame(result.get("residuals"), columns=["Residuos del Modelo"])

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
residual['Residuos del Modelo'].plot(ax=axs[0, 0])
axs[0, 0].set_title("Residuos")
axs[0, 0].set_xlabel('Tiempo')

sns.histplot(residual['Residuos del Modelo'], ax=axs[0, 1])
axs[0, 1].set_title("Densidad - Residuos")

stats.probplot(residual['Residuos del Modelo'], dist="norm", plot=axs[1, 0])
axs[1, 0].set_title('Gráfico Q-Q')

plot_acf(residual['Residuos del Modelo'], lags=35, ax=axs[1, 1], color="blue")
axs[1, 1].set_title("Autocorrelación de Residuos")
plt.tight_layout()
plt.show()

# Generación y visualización de previsiones
Y_hat_df = sf.forecast(horizon, fitted=True)
print(Y_hat_df.head())

forecast_df = sf.forecast(h=28, level=[95])

df_plot = pd.concat([df_selected, forecast_df]).set_index('ds').tail(75)

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
plt.plot(df_plot.index, df_plot['y'], 'k--', label="Actual", linewidth=2)
plt.plot(df_plot.index, df_plot['AutoARIMA'], label="Previsión AutoARIMA", linewidth=2, color="red")
ax.fill_between(df_plot.index,
                df_plot['AutoARIMA-lo-95'],
                df_plot['AutoARIMA-hi-95'],
                alpha=.2,
                color='red',
                label='Intervalo de Confianza 95%')

ax.set_title('Previsión diaria del precio de Bitcoin', fontsize=20)
ax.set_ylabel('Precio BTC', fontsize=15)
ax.set_xlabel('Fecha', fontsize=15)
ax.legend(prop={'size': 12})
ax.grid(True)

plt.tight_layout()
plt.show()
