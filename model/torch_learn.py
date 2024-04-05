import torch
from torch_infer import infer
from torch_functions import f

#learning is *almost* same except we use localised errors and not global errors
def learn(in_data,out_data,w,b,params):
  n_layers = params["n_layers"] #no layers +1
  l_rate = params["l_rate"]
  variance = params["variance"]
  itr_max = params["itr"]
  n_layers = params["n_layers"]
  beta = params["beta"]
  neurons = params["neurons"]
  act_type = params["act_type"]
  alpha = params["alpha"]
  iterations = len(in_data)
  v_out = variance[-1]


  for i in range(iterations):
    x = [None] * n_layers
    grad_w = [None] * len(w)
    grad_b = [None] * len(b)

    x[0] = in_data[i].reshape(1,-1)
    x_out = out_data[i].reshape(1,-1)

    for j in range(1,n_layers):
      x[j] = (f(x[j-1].reshape(1,-1), act_type, alpha) @ w[j-1].T ) + b[j-1].T

    x[-1]  = x_out
    x, e = infer(x,w,b,params)

    for j in range(n_layers-1):
      grad_b[j] = v_out * e[j+1].T
      grad_w[j] = v_out * e[j+1].T  @  f(x[j], act_type, alpha)

    for j in range(n_layers-1):
      w[j] += l_rate * grad_w[j]
      b[j] += l_rate * grad_b[j]
