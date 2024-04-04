import torch
from torch_functions import f_b

def infer(x,w,b,params):
  itr_max = params["itr"]
  n_layers = params["n_layers"]
  beta = params["beta"]
  variance = params["variance"]
  neurons = params["neurons"]
  act_type = params["act_type"]
  alpha = params["alpha"]

  e = [None] * (n_layers)
  f_n = [None] * (n_layers)
  f_p = [None] * (n_layers)

  for i in range(n_layers):
    f_n[i], f_p[i] = f_b(x[i], act_type, alpha)
    if i == 0:
      e[i] = torch.zeros((1,neurons[0]))
    else:
      e[i] = (x[i] - ((f_n[i-1] @ w[i-1].T) + b[i-1].T))/variance[i]

  for j in range(itr_max):
    for i in range(1,n_layers-1):
      g = (w[i].T @ e[i+1].T).T * f_p[i]
      x[i] = x[i] + beta * (- e[i] + g)
    for i in range(n_layers):
      f_n[i], f_p[i] = f_b(x[i], act_type, alpha)
      if i == 0:
        e[i] = torch.tensor([0], dtype=torch.float32).reshape(1,-1)
      else:
        e[i] = (x[i] - ((f_n[i-1] @ w[i-1].T) + b[i-1].T))/variance[i]
  return x, e