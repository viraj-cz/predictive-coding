import torch

def xavier_init_tanh(neurons):
  n_layers = len(neurons)
  w = [None] * (n_layers - 1)
  b = [None] * (n_layers - 1)

  for i in range(n_layers - 1):
    norm_w = torch.sqrt(torch.tensor(6 / (neurons[i + 1] + neurons[i])))
    #why is the one below incorrect?
    #w[i] = (-1 - 1) * torch.rand((neurons[i + 1], neurons[i])) * norm_w + 1
    w[i] = torch.rand((neurons[i + 1], neurons[i])) * 2 * norm_w - norm_w

    b[i] = torch.zeros((neurons[i + 1], 1))
  return w, b

def he_init_relu(neurons):
    n_layers = len(neurons)
    w = [None] * (n_layers - 1)
    b = [None] * (n_layers - 1)

    for i in range(n_layers - 1):
        norm_w = torch.sqrt(torch.tensor(2 / neurons[i]))
        w[i] = torch.randn(neurons[i + 1], neurons[i]) * norm_w
        b[i] = torch.zeros((neurons[i + 1], 1))
    return w, b

def std_gaussian(neurons):
    n_layers = len(neurons)
    w = [None] * (n_layers - 1)
    b = [None] * (n_layers - 1)

    for i in range(n_layers - 1):
        w[i] = torch.randn(neurons[i + 1], neurons[i])
        b[i] = torch.zeros((neurons[i + 1], 1))
    return w, b

def uniform(neurons):
    n_layers = len(neurons)
    w = [None] * (n_layers - 1)
    b = [None] * (n_layers - 1)

    for i in range(n_layers - 1):
        w[i] = torch.rand(neurons[i + 1], neurons[i])
        b[i] = torch.zeros((neurons[i + 1], 1))
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