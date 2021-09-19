import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch import nn


class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.y = torch.tensor(self.data['target'].values.astype(np.float32))
        self.x = torch.tensor(self.data.loc[:, self.data.columns != 'target'].values.astype(np.float32))

    def __getitem__(self, index):
        X, Y = self.x[index], self.y[index]
        return X, Y

    def __len__(self):
        return (len(self.x))


dataset = CustomDataset("/home/surya/PycharmProjects/clevland_heart_disease/heart.csv")

batch_size = 16
test_split = 0.2
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Network Parameters

input_size = 13
hidden_size = [56, 56]
output_size = 1

device = "cuda"


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(input_size, hidden_size[0])
        self.output = nn.Linear(hidden_size[1], output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


network = Network()
network.cuda()

criterion = nn.BCELoss()
criterion.cuda()

optimizer = torch.optim.Adam(network.parameters(), lr=0.01)


def train(dataloader, network, criterion, optimizer):
    size = len(dataloader)
    network.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y = y.unsqueeze(1)
        pred = network(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.unsqueeze(1)
            pred = network(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 500
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, network, criterion, optimizer)
    test(test_loader, network, criterion)
print("Done!")