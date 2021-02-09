import pandas as pd
import torch
import torch.nn as nn
# import dataset and get inputs and targets
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F

df = pd.read_csv('./data/winequality-red.csv', delimiter=';', skiprows=1, header=None)
df_input = df.iloc[:, :11]
df_output = df.iloc[:, -1:]

input_np_array = df_input.to_numpy()
output_np_array = df_output.to_numpy()

input = torch.from_numpy(input_np_array)
target = torch.from_numpy(output_np_array)

df_all_data = df.to_numpy()
df_all_data_tensor = torch.from_numpy(df_all_data)


# to load data and split into training and test data
def load_data(test_split, batch_size):
    """:cvar
    """
    dataset_size = len(df_all_data)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = random_split(df_all_data, [train_size, test_size])
    train_loader = DataLoader(train_dataset.dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset.dataset, batch_size=batch_size, shuffle=True)
    print('>>>> ', type(train_loader))

    return train_loader, test_loader


train_dl, test_dl = load_data(0.2, 32)

# Define model

model = nn.Linear(11, 1)
# print(model.weight)
# print(model.bias)
# model parameters
loss_fx = F.mse_loss
#SGD defining optimiser
opt = torch.optim.SGD(model.parameters(),lr=1e-5)

for (batch_idx, batch) in enumerate(train_dl):
    X = batch[:, :11]
    Y = batch[:, -1:]
    #1. Geneerate predictions
    predictions = model(X.float())
    #2. Calculate loss
    loss = loss_fx(model(X.float()), Y.float())
    #3. Compute gradients
    loss.backward()
    #4. Update parameters using gradients
    opt.step()
    #5. Reset the gradients to zero
    opt.zero_grad()
    print("Batch = " + str(batch_idx) +" loss: "+ str(loss.item()))
