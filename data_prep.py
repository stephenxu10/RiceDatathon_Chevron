import torch
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_train_data(overall_data, states):
    input_data = torch.ones(4 * len(states), 29, device=device)
    predictions = torch.ones(4 * len(states), device=device)

    states_data = overall_data.loc[(overall_data['StateCode']).isin(states)]

    for i in range(4):
        year_data = states_data.loc[states_data['Year'] == 2015 + i]
        for j in range(len(states)):
            predictions[i * len(states) + j] = torch.tensor(year_data.iat[j * 29, 8])

            for k in range(29):
                input_data[i * len(states) + j][k] = torch.tensor(year_data.iat[k + j * 29, 4])

    predictions = torch.reshape(predictions, (4, len(states)))

    return input_data, predictions


def prepare_test_data(overall_data, states):
    input_data = torch.ones(len(states), 29, device=device)
    predictions = torch.ones(len(states), device=device)

    states_data = overall_data.loc[(overall_data['StateCode']).isin(states)]

    year_data = states_data.loc[states_data['Year'] == 2019]
    for j in range(len(states)):
        predictions[j] = torch.tensor(year_data.iat[j * 29, 8])

        for k in range(29):
            input_data[j][k] = torch.tensor(year_data.iat[k + j * 29, 4])

    predictions = torch.reshape(predictions, (1, len(states)))

    return input_data, predictions



