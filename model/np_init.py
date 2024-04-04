import numpy as np

def xavier_init_tanh(neurons):
  n_layers = len(neurons)
  w = [None] * (n_layers - 1)
  b = [None] * (n_layers - 1)

  for i in range(n_layers - 1):
      norm_w = np.sqrt(6 / (neurons[i + 1] +neurons[i]))
      w[i] = np.random.uniform(-1, 1, (neurons[i + 1], neurons[i])) * norm_w
      b[i] = np.zeros((neurons[i + 1], 1))
  return w, b

def he_init_relu(neurons):
    n_layers = len(neurons)
    w = [None] * (n_layers - 1)
    b = [None] * (n_layers - 1)

    for i in range(n_layers - 1):
        norm_w = np.sqrt(2 / neurons[i])
        w[i] = np.random.randn(neurons[i + 1], neurons[i]) * norm_w
        b[i] = np.zeros((neurons[i + 1], 1))
    return w, b

def std_gaussian(neurons):
    n_layers = len(neurons)
    w = [None] * (n_layers - 1)
    b = [None] * (n_layers - 1)

    for i in range(n_layers - 1):
        w[i] = np.random.randn(neurons[i + 1], neurons[i])
        b[i] = np.zeros((neurons[i + 1], 1))
    return w, b

def uniform(neurons):
    n_layers = len(neurons)
    w = [None] * (n_layers - 1)
    b = [None] * (n_layers - 1)

    for i in range(n_layers - 1):
        w[i] = np.random.uniform(-1,1,(neurons[i + 1], neurons[i]))
        b[i] = np.zeros((neurons[i + 1], 1))
    return w, b

def initializer(neurons, picker):
    if picker == "xavier":
        w, b = xavier_init_tanh(neurons)
    elif picker == "he":
        w, b = he_init_relu(neurons)
    elif picker == "std_gaussian":
        w, b = std_gaussian(neurons)
    elif picker == "uniform":
        w, b = uniform(neurons)
    else:
        raise Exception("wrong activation type")
    return w, b