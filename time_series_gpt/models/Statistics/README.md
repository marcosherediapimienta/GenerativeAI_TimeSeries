## Introducción

En esta lección, descubrirás una forma específica de construir modelos con [ARIMA: Promedio Móvil Integrado AutoRegresivo](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average/ "ARIMA: Promedio Móvil Integrado AutoRegresivo"). Los modelos ARIMA son particularmente adecuados para ajustar datos que muestran [no estacionariedad](https://en.wikipedia.org/wiki/Stationary_process/ "no estacionariedad").

## Conceptos Generales

Para poder trabajar con ARIMA, hay algunos conceptos que necesitas conocer:

**Estacionariedad** Desde un contexto estadístico, la estacionariedad se refiere a datos cuya distribución no cambia cuando se desplazan en el tiempo. Los datos no estacionarios, por lo tanto, muestran fluctuaciones debido a tendencias que deben ser transformadas para ser analizadas. La estacionalidad, por ejemplo, puede introducir fluctuaciones en los datos y puede ser eliminada mediante un proceso de 'diferenciación estacional'.

**[Diferenciación](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing/ "Diferenciación")** Diferenciar los datos, nuevamente desde un contexto estadístico, se refiere al proceso de transformar datos no estacionarios para hacerlos estacionarios eliminando su tendencia no constante. "La diferenciación elimina los cambios en el nivel de una serie temporal, eliminando la tendencia y la estacionalidad y, por consiguiente, estabilizando la media de la serie temporal." [Paper by Shixiong et al](https://arxiv.org/abs/1904.07632/ "Paper by Shixiong et al")

## ARIMA en el contexto de series temporales

Desglosaremos las partes de ARIMA para entender mejor cómo nos ayuda a modelar series temporales y hacer predicciones.

AR - de AutoRegresivo. Los modelos autorregresivos, como su nombre indica, miran "hacia atrás" en el tiempo para analizar valores anteriores en tus datos y hacer suposiciones sobre ellos. Estos valores anteriores se llaman "rezagos". Un ejemplo serían los datos que muestran las ventas mensuales de lápices. El total de ventas de cada mes se consideraría una "variable en evolución" en el conjunto de datos. Este modelo se construye como "la variable de interés en evolución se regresa sobre sus propios valores rezagados (es decir, anteriores)." (Wikipedia)

I - de Integrado. A diferencia de los modelos similares 'ARMA', la 'I' en ARIMA se refiere a su aspecto [integrado](https://en.wikipedia.org/wiki/Order_of_integration/ "integrado"). Los datos se "integran" cuando se aplican pasos de diferenciación para eliminar la no estacionariedad.

MA - de Promedio Móvil. El aspecto de [promedio móvil](https://en.wikipedia.org/wiki/Moving-average_model/ "promedio móvil") de este modelo se refiere a la variable de salida que se determina observando los valores actuales y pasados de los rezagos.