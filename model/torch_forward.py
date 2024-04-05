from torch_functions import f

def predict(in_data_single,w,b,n_layers, act_type, alpha):
  x = [None] * n_layers
  x[0] = in_data_single
  for j in range(1,n_layers):
    x[j] = (f((x[j-1].reshape(1,-1)), act_type, alpha) @ w[j-1].T) + b[j-1].T
  predicted = x[-1]
  return predicted[0][0]