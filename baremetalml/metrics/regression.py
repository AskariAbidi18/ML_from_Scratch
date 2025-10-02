import numpy as np
def MAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / len(y_true)

def MSE(y_true, y_pred):
    return np.sum((y_true - y_pred) **2) / len(y_true)

def RMSE(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) **2) / len(y_true))

def R2(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
