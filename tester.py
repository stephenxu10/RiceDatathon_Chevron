import torch
from torch import nn
import pandas as pd
import model, data_prep
from model import NeuralNetwork

device = 'cuda'

overall_data = pd.read_csv("./chevron.csv")

states = ['TX']

inputs, outputs = data_prep.prepare_train_data(overall_data, states)
test_inputs, test_outputs = data_prep.prepare_test_data(overall_data, states)
norms = inputs.norm(dim=1)


print("===================")


print(test_inputs)
print(test_outputs)

inputs = torch.nn.functional.normalize(inputs, dim=0)
outputs = torch.nn.functional.normalize(outputs, dim=1)

nn_model = NeuralNetwork(len(states)).to(device)

epochs = 100
l_rate = 0.01
loss_fn = torch.nn.MSELoss()

model.train_loop(nn_model, inputs, outputs, loss_fn, l_rate, epochs)

print(nn_model(test_inputs) * norms.item())