import numpy as np


def one_hot_encoding(y):
    new_y = []
    for label in y:
        label_vector = np.zeros(20)
        label_vector[label - 1] = 1
        new_y.append(label_vector)
    y = np.array(new_y)
    return y


def shuffle(X, y):
    index = np.arange(np.shape(X)[1])
    np.random.shuffle(index)
    X = X[:, index]
    y = y.T[:, index]
    return X, y


def split_data(X, y, split_value=0.8):
    m = X.shape[1]
    m = int(m * split_value)
    X_train = X[:, :m]
    y_train = y[:m, :]
    X_test = X[:, m:]
    y_test = y[m:, :]
    return X_train, y_train, X_test, y_test


def loss_function(y, y_pred):
    n = y.shape[1]
    loss = (np.linalg.norm(y - y_pred, 'fro') ** 2) / n
    return loss

def least_square(a, b):
    return 0
