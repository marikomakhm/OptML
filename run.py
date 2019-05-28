import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale

from mlp import MLP
from helpers import load_data

def train(model, train_loader, optimizer, criterion=nn.NLLLoss(), nb_epochs=10, verbose=True):
    train_losses = []
    train_counter = []
    for epoch in range(nb_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*batch_size) + (epoch*len(train_loader.dataset)))
    return train_losses, train_counter

def train_average_sgd(train_loader, nb_models=10, nb_epochs=10):
    all_train_losses = []
    train_counter = []
    for i in range(nb_models):
        print('Training model %d...' % i)
        model = MLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        train_losses, train_counter = train(model, train_loader, optimizer, nb_epochs)
        all_train_losses.append(train_losses)
    return np.mean(all_train_losses, axis=0), train_counter

def train_average_adam(train_loader, nb_models=10, nb_epochs=10):
    all_train_losses = []
    train_counter = []
    for i in range(nb_models):
        print('Training model %d...' % i)
        model = MLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, momentum=0.9)
        train_losses, train_counter = train(model, train_loader, optimizer, nb_epochs)
        all_train_losses.append(train_losses)
    return np.mean(all_train_losses, axis=0), train_counter

batch_size = 64

train, test = load_data('data/train.csv', test_size=0.2, standardize=True)

# data loaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

