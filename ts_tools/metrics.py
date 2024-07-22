import numpy as np

# Función para calcular SMAPE
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Función para calcular MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Función para calcular MAE manualmente
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Función para calcular RMSE manualmente
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))