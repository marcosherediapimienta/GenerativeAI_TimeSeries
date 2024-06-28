import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv("/Users/marcosherediapimienta/Library/Mobile Documents/com~apple~CloudDocs/Documents/Máster de Matemàtiques per els Instruments Financers/TFM/Time_Series/archive/Top10-2021-2024-1d.csv")

# Seleccionar las columnas deseadas del DataFrame original
df_selected = df[['Timestamp', 'BTCUSDT']]
df_selected = df_selected.rename(columns={'Timestamp': 'ds', 'BTCUSDT': 'y'})
df_selected = df_selected.dropna()

# Agregar la columna unique_id
df_selected["unique_id"] = "1"

# Convertir la columna ds a tipo datetime
df_selected["ds"] = pd.to_datetime(df_selected["ds"])

# Seleccionar la columna de valores numéricos para la función de autocorrelación
btc_values = df_selected['y']

plt.figure(figsize=(14, 7))

# Graficar la serie temporal del BTC
plt.subplot(1, 2, 1)
plt.plot(df_selected['ds'], df_selected['y'])
plt.title('Serie Temporal de BTC')
plt.xlabel('Fecha')
plt.ylabel('Valor')

# Graficar la función de autocorrelación
plt.subplot(1, 2, 2)
plot_acf(btc_values, lags=20, ax=plt.gca(), alpha=0.05)
plt.ylim(-1, 1)
plt.title('Autocorrelación Simple')

# Ajustar el diseño de los gráficos y mostrar
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Ajustar el espacio entre los subgráficos
plt.show()
