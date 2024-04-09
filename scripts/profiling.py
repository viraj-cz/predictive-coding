#scalene --profile-all scripts/profiling.py
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
model_path = os.path.join(os.path.dirname(__file__), '..', 'model')
sys.path.insert(0, model_path)
# import np_functions
# import np_init
# import np_loss
# import np_infer
# import np_learn
# import np_forward
import torch_functions
import torch_init
import torch_loss
import torch_infer
import torch_learn
import torch_forward

parameters = {
    "itr":100,
    "l_rate":0.2, 
    "epochs":500,
    "beta":0.2,
    "act_type": "RELU",
    "alpha": 1, 
    "neurons": [2,3,1], 
    "variance": [1,1,10]
}
parameters["n_layers"] = (len(parameters["neurons"]) ) #basically how many layers we have + 1

in_data = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
out_data = torch.tensor([1,0,0,1], dtype=torch.float32)
w,b = torch_init.initializer(parameters["neurons"], "xavier")

def train():
    for i in range(parameters["epochs"]):
        torch_learn.learn(in_data,out_data, w ,b, parameters)

train()
