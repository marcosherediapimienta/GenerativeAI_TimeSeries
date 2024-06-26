## Intrucción

En esta lección, descubrirás una forma específica de construir modelos con ARIMA: Promedio Móvil Integrado AutoRegresivo. Los modelos ARIMA son particularmente adecuados para ajustar datos que muestran no estacionariedad.

## Conceptos Generales

Para poder trabajar con ARIMA, hay algunos conceptos que necesitas conocer:

**Estacionariedad** Desde un contexto estadístico, la estacionariedad se refiere a datos cuya distribución no cambia cuando se desplazan en el tiempo. Los datos no estacionarios, por lo tanto, muestran fluctuaciones debido a tendencias que deben ser transformadas para ser analizadas. La estacionalidad, por ejemplo, puede introducir fluctuaciones en los datos y puede ser eliminada mediante un proceso de 'diferenciación estacional'.

**Diferenciación** Diferenciar los datos, nuevamente desde un contexto estadístico, se refiere al proceso de transformar datos no estacionarios para hacerlos estacionarios eliminando su tendencia no constante. "La diferenciación elimina los cambios en el nivel de una serie temporal, eliminando la tendencia y la estacionalidad y, por consiguiente, estabilizando la media de la serie temporal." [Paper by Shixiong et al](https://arxiv.org/abs/1904.07632/ "Paper by Shixiong et al")