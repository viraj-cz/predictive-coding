import numpy as np
'''
f_n = activation(x)
f_b = derivative of f_n wrt x
'''

def f_b_relu(x):
  f_n = np.maximum(x,0)
  f_p = (x > 0).astype(x.dtype)
  return f_n, f_p

def f_b_tanh(x):
  f_n = np.tanh(x)
  f_p = (1 - (np.tanh(x) ** 2))
  return f_n, f_p

def f_b_sigmoid(x, alpha):
  f_n = 1/(1 + np.exp(-1 * alpha * x))
  f_p = f_n * (1 - f_n)
  return f_n, f_p

def f(x, act_type, alpha):
  if act_type == "RELU":
    f_n, f_p = f_b_relu(x)
    return f_n
  elif act_type == "TANH":
    f_n, f_p = f_b_tanh(x)
    return f_n
  elif act_type == "SIGMOID":
    f_n, f_p = f_b_sigmoid(x, alpha)
    return f_n
  else:
    raise("invalid actiavtion function")

def f_b(x, act_type, alpha):
  if act_type == "RELU":
    f_n, f_p = f_b_relu(x)
    return f_n, f_p
  elif act_type == "TANH":
    f_n, f_p = f_b_tanh(x)
    return f_n, f_p
  elif act_type == "SIGMOID":
    f_n, f_p = f_b_sigmoid(x,alpha)
    return f_n, f_p
  else:
    raise("invalid actiavtion function")