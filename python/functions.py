import numpy as np
import csv


def one_hot_encoding(y):
    new_y = []
    for label in y:
        label_vector = np.zeros(20)
        label_vector[label - 1] = 1
        new_y.append(label_vector)
    y = np.array(new_y)
    return y


def reverse_one_hot_encoding(y):
    y = np.argmax(y, axis=0) + 1
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


def grad_W(W, B, X, y, n):
    return X @ ((1 / n) * np.multiply(sigmoid(W.T @ X + B, 1) - y, sigmoid_derivate(W.T @ X + B, 1))).T


def grad_B(W, B, X, y, n):
    v1 = np.ones((n, 1))
    return ((1 / n) * np.multiply(sigmoid(W.T @ X + B, 1) - y, sigmoid_derivate(W.T @ X + B, 1))) @ v1


def sigmoid(X, a):
    return 1 / (1 + np.exp(-a * X))


def sigmoid_derivate(X, a):
    return sigmoid(X, a) * (1 - sigmoid(X, a))


def relu(x, a):
    return np.maximum(0, a * x)


def test(X, W, b):
    _, n = X.shape
    B = np.tile(b, (n, 1)).T
    y = sigmoid(W.T @ X + B, 1)
    y = reverse_one_hot_encoding(y)
    ids = np.arange(1, len(y) + 1)
    data = np.column_stack((ids, 1 + 100 * y))
    nom_fichier = "../results/test.csv"

    header = ["id", "class"]
    with open(nom_fichier, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)