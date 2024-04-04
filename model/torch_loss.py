import torch

'''
inputs:
y,f both tensors
returns loss, wrapped as a tensor
'''

def RMS_error(y, f):
    rmse = torch.sqrt(torch.mean((y - f) ** 2))
    return rmse

def MSE_error(y, f):
    mse = (torch.mean((y - f) ** 2))
    return mse