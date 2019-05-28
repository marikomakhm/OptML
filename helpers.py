import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale

def load_data(filepath, test_size=0.2, standardize=True):
    df = pd.read_csv(filepath)
    y = df['label'].values
    X = df.drop(['label'],1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if standardize:
        X_train = scale(X_train, axis=1, with_mean=True, with_std=True)
        X_test = scale(X_test, axis=1, with_mean=True, with_std=True)

    # cast to pytorch tensor
    torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
    torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    # pytorch train and test sets
    train = TensorDataset(torch_X_train, torch_y_train)
    test = TensorDataset(torch_X_test, torch_y_test)

    return train, test