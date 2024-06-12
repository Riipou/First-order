import numpy as np
import csv
from scipy.io import loadmat


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


def split_data(X, y, split_value=0.95):
    m = X.shape[1]
    m = int(m * split_value)
    X_train = X[:, :m]
    y_train = y[:, :m]
    X_test = X[:, m:]
    y_test = y[:, m:]
    return X_train, y_train, X_test, y_test


def loss_function(y, y_pred):
    n = y.shape[1]
    loss = (np.linalg.norm(y - y_pred, 'fro') ** 2) / (2 * n)
    return loss


def loss_function_L2(y, y_pred, lbd, W):
    n = y.shape[1]
    loss = ((np.linalg.norm(y - y_pred, 'fro') ** 2) / (2 * n)) + (lbd / 2) * np.sum(W ** 2)
    return loss


def sigmoid(X, a):
    return 1 / (1 + np.exp(-a * X))


def sigmoid_derivate(X, a):
    return a * sigmoid(X, a) * (1 - sigmoid(X, a))


def relu(x, a):
    return np.maximum(0, a * x)


def initialization_random(m, p):
    W = np.random.rand(m, p) - 0.5
    b = np.random.rand(p) - 0.5
    return W, b


def initialization_xavier(m, p):
    W = np.random.randn(m, p) * np.sqrt(2 / (m + p))
    b = np.zeros((p,))
    return W, b


def initialization_lecun(m, p):
    W = np.random.randn(m, p) * np.sqrt(1 / m)
    b = np.zeros((p,))
    return W, b


def grad_W(W, B, a, X, y):
    _, n = X.shape
    _, p = W.shape
    if B.shape == (p,):
        # Reshape to (p, 1)
        B = B.reshape(p, 1)

    if y.shape == (p,):
        # Reshape to (p, 1)
        y = y.reshape(p, 1)

    return X @ ((1 / n) * np.multiply(sigmoid(W.T @ X + B, a) - y, sigmoid_derivate(W.T @ X + B, a))).T


def grad_W_L2(W, B, a, X, y, lbd):
    _, n = X.shape
    _, p = W.shape
    if B.shape == (p,):
        # Reshape to (p, 1)
        B = B.reshape(p, 1)

    if y.shape == (p,):
        # Reshape to (p, 1)
        y = y.reshape(p, 1)

    return X @ ((1 / n) * np.multiply(sigmoid(W.T @ X + B, a) - y, sigmoid_derivate(W.T @ X + B, a))).T + lbd * W


def grad_B(W, B, a, X, y):
    _, n = X.shape
    _, p = W.shape
    v1 = np.ones((n, 1))
    if B.shape == (p,):
        # Reshape to (p, 1)
        B = B.reshape(p, 1)

    if y.shape == (p,):
        # Reshape to (p, 1)
        y = y.reshape(p, 1)
    return ((1 / n) * np.multiply(sigmoid(W.T @ X + B, a) - y, sigmoid_derivate(W.T @ X + B, a))) @ v1


def evol_beta_paul(k):
    return (k - 1) / (k + 2)


def evol_beta_nesterov(alpha_b):
    alpha_b2 = (np.sqrt((alpha_b ** 4) + 4 * alpha_b ** 2) - alpha_b ** 2) / 2
    beta = (alpha_b * (1 - alpha_b)) / (alpha_b ** 2 + alpha_b2)
    return beta, alpha_b2


def test(X, W, b, nb_ite, a, stop, init, accel, evol_beta):
    _, n = X.shape
    B = np.tile(b, (n, 1)).T
    y = sigmoid(W.T @ X + B, 1)
    y = reverse_one_hot_encoding(y)
    ids = np.arange(1, len(y) + 1)
    data = np.column_stack((ids, 1 + 100 * y))
    if accel != "none":
        nom_fichier = f"../results/tests/test_{nb_ite}_ite_sigmoid_{a}_stop_{stop}_{init}_{accel}_beta_{evol_beta}.csv"
    else:
        nom_fichier = f"../results/tests/test_{nb_ite}_ite_sigmoid_{a}_stop_{stop}_{init}.csv"

    header = ["id", "class"]
    with open(nom_fichier, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)


def load_mat_file(matfile):
    donnees = loadmat(matfile)

    # Xts combien de fois le mot dans le texte
    if 'Xts' in donnees:
        Xts = donnees['Xts']
    else:
        print("La variable 'Xts' n'a pas été trouvée dans le fichier.")

    # id texte avec classe
    if 'yts' in donnees:
        yts = donnees['yts']
    else:
        print("La variable 'yts' n'a pas été trouvée dans le fichier.")

    # Xvr combien de fois le mot dans le texte
    if 'Xvr' in donnees:
        Xvr = donnees['Xvr']
    else:
        print("La variable 'Xvr' n'a pas été trouvée dans le fichier.")

    return Xts, yts, Xvr
