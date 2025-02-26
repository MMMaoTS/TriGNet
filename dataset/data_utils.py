import numpy as np
from scipy.io import loadmat
import torch


def prepare_data_mitpsg():
    print("[MIT-PSG data] Loading data...")
    X = loadmat("./dataset/mitpsg/pre_data.mat")["x"]
    Y = loadmat("./dataset/mitpsg/pre_label.mat")["y"]
    M = len(X)

    N1 = int(0.6 * M)
    N2 = int(0.8 * M)

    x_train = np.expand_dims(X[:N1], axis=1)
    y_train = Y[:N1]
    x_val = np.expand_dims(X[N1:N2], axis=1)
    y_val = Y[N1:N2]
    x_test = np.expand_dims(X[N2:], axis=1)
    y_test = Y[N2:]

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train = [x_train, y_train]
    val = [x_val, y_val]
    test = [x_test, y_test]
    print("[MIT-PSG] Done")
    return train, val, test

def prepare_data_mitpsgbi():
    print("[MIT-PSG bi data] Loading data...")
    X = loadmat("./dataset/mitpsg/pre_bi_data.mat")["x"]
    Y = loadmat("./dataset/mitpsg/pre_bi_label.mat")["y"]
    M = len(X)

    N1 = int(0.6 * M)
    N2 = int(0.8 * M)

    x_train = np.expand_dims(X[:N1], axis=1)
    y_train = Y[:N1]
    x_val = np.expand_dims(X[N1:N2], axis=1)
    y_val = Y[N1:N2]
    x_test = np.expand_dims(X[N2:], axis=1)
    y_test = Y[N2:]

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train = [x_train, y_train]
    val = [x_val, y_val]
    test = [x_test, y_test]
    print("[MIT-PSG bi] Done")
    return train, val, test


def prepare_data_ae():
    print("[Apnea-ECG] Loading data...")
    x_train = loadmat("./dataset/ae/ae-train.mat")["x"]
    y_train = loadmat("./dataset/ae/ae-train.mat")["y"] 
    x_val = loadmat("./dataset/ae/ae-val.mat")["x"]
    y_val = loadmat("./dataset/ae/ae-val.mat")["y"]
    x_test = loadmat("./dataset/ae/ae-test.mat")["x"]
    y_test = loadmat("./dataset/ae/ae-test.mat")["y"]

    x_train = np.expand_dims(x_train, axis=1)
    x_val = np.expand_dims(x_val, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train = [x_train, y_train]
    val = [x_val, y_val]
    test = [x_test, y_test]
    print("[Apnea-ECG] Done")
    return train, val, test
