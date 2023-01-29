import torch
from torch import nn
import pandas as pd
import model, data_prep
from model import NeuralNetwork


device = 'cuda'

training_data = pd.read_csv("./data/flipped.csv")
test_data = pd.read_csv("./data/2020flipped.csv")

states = training_data['State'].unique()

inputs, outputs = data_prep.prepare_train_data(training_data, states)
test_inputs, test_outputs = data_prep.prepare_test_data(test_data)
#norms = inputs.norm(dim=1)


test_inputs = test_inputs.float().cuda()
print(test_inputs.shape)
print(test_inputs)
test_outputs = test_outputs.float().cuda()

inputs = torch.nn.functional.normalize(inputs, dim=2)
outputs = torch.nn.functional.normalize(outputs, dim=0)
print("======")

print(test_inputs.shape)


print("===================")

#inputs = torch.nn.functional.normalize(inputs, dim=0)
#outputs = torch.nn.functional.normalize(outputs, dim=1)

nn_model = NeuralNetwork(len(states)).to(device)
nn_model = nn_model.float()

epochs = 500
l_rate = 0.005
loss_fn = torch.nn.MSELoss()

print(inputs)
model.train_loop(nn_model, inputs, outputs, loss_fn, l_rate, epochs)

print(test_inputs)
print(nn_model(test_inputs))
