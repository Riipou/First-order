from functions import *
from scipy.io import loadmat
import numpy as np
import csv


def update_W(alpha, W, B, X, y):
    loss = loss_function(y, sigmoid(W.T @ X + B, 1))
    alpha *= 1.5
    W2 = W - alpha * grad_W(W, B, X, y, n)
    while loss_function(y, sigmoid(W2.T @ X + B, 1)) > loss:
        alpha /= 2
        W2 = W - alpha * grad_W(W, B, X, y, n)
    return W2, alpha


def update_B(alpha, W, B, X, y):
    loss = loss_function(y, sigmoid(W.T @ X + B, 1))
    alpha *= 1.5
    b = B[:, 1]
    b = b.reshape(-1, 1)
    b2 = b - alpha * grad_B(W, B, X, y, n)
    B2 = np.tile(b2.reshape(-1), (n, 1)).T
    while loss_function(y, sigmoid(W.T @ X + B2, 1)) > loss:
        alpha /= 2
        b2 = b - alpha * grad_B(W, B, X, y, n)
        B2 = np.tile(b2.reshape(-1), (n, 1)).T
    return B2, alpha


def linear_model(m, n, p, max_iterations, X, y):
    W = np.random.rand(m, p) - 0.5
    b = np.random.rand(p) - 0.5
    B = np.tile(b, (n, 1)).T
    alpha_W = 0.1 * np.linalg.norm(W, 'fro') / np.linalg.norm(grad_W(W, B, X, y, n), 'fro')
    alpha_B = 0.1 * np.linalg.norm(B, 'fro') / np.linalg.norm(grad_B(W, B, X, y, n), 'fro')
    for _ in range(max_iterations):
        loss_prev = loss_function(y, sigmoid(W.T @ X + B, 1))
        W, alpha_W = update_W(alpha_W, W, B, X, y)
        B, alpha_B = update_B(alpha_B, W, B, X, y)
        loss = loss_function(y, sigmoid(W.T @ X + B, 1))
        print(loss)
        if abs(loss - loss_prev) < 1e-5:
            print("break")
            break
    return W, b


def test(X, W, b):
    _, n = X.shape
    B = np.tile(b, (n, 1)).T
    y = sigmoid(W.T @ X + B, 1)
    y = reverse_one_hot_encoding(y)
    ids = np.arange(1, len(y)+1)
    data = np.column_stack((ids, 1+100 * y))
    nom_fichier = "test2.csv"

    header = ["id","class"]
    with open(nom_fichier, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)
    return 0


if __name__ == "__main__":

    # Charger le fichier .mat
    nom_fichier_mat = 'data_doc.mat'
    donnees = loadmat(nom_fichier_mat)

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

    yts = one_hot_encoding(yts)
    Xts, yts = shuffle(Xts, yts)
    # X_train, y_train, X_test, y_test = split_data(Xts, yts)
    m, n = Xts.shape
    p, _ = yts.shape
    W, b = linear_model(m, n, p, 100000, Xts, yts)
    test(Xvr, W, b)
