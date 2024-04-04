import numpy as np

def RMS_error(y, f):
    rmse = np.sqrt(np.mean((y - f) ** 2))
    return rmse

def MSE_error(y, f):
    mse = (np.mean((y - f) ** 2))
    return mse