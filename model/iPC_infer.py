import numpy as np
from np_functions import f_b

#this is the heart of pred code
def infer(x,w,b,params):
  itr_max = params["itr"]
  n_layers = params["n_layers"]
  beta = params["beta"]
  variance = params["variance"]
  neurons = params["neurons"]
  act_type = params["act_type"]
  alpha = params["alpha"]

  e = [None] * n_layers
  f_n = [None] * (n_layers)
  f_p = [None] * (n_layers)

  for i in range(n_layers):
    f_n[i], f_p[i] = f_b(x[i], act_type, alpha)
    if i == 0:
      e[i] = np.zeros((1,neurons[0]))
    else:
      e[i] = (x[i] - ((f_n[i-1] @ w[i-1].T) + b[i-1].T))/variance[i]

  for j in range(itr_max):
    for i in range(1,n_layers-1):
      #this algorithm is not perfect and not optimized for computations but it works
      #TODO: make it perfect later
      rand_idx = np.random.choice(len(x[i][0]))
      g = (w[i].T @ e[i+1].T).T * f_p[i]
      new_x = x[i] + beta * (- e[i] + g)
      #print(f"idx = {rand_idx}")
      #print(f"pre_x { x[i][0]}")
      #print(f"new_x = {new_x}")
      x[i][0][rand_idx] = new_x[0][rand_idx]
      #print(f"post_x { x[i][0]}")
    for i in range(n_layers):
      f_n[i], f_p[i] = f_b(x[i], act_type, alpha)
      if i == 0:
        e[i] = np.array([0]).reshape(1,-1)
      else:
        e[i] = (x[i] - ((f_n[i-1] @ w[i-1].T) + b[i-1].T))/variance[i]
  return x, e
