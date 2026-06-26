"""Regression metrics used in the paper: MAD/MAE, RMSE, MSE, MAPE."""
from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def performance_metrics(y_true, y_pred):
    """Return (MAE/MAD, RMSE, MSE, MAPE)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mse, mape
