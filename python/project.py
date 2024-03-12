from functions import *
from scipy.io import loadmat
import numpy as np


def update_W(alpha, W, B, a, X, y):
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5
    W2 = W - alpha * grad_W(W, B, a, X, y, n)
    while loss_function(y, sigmoid(W2.T @ X + B, a)) > loss:
        alpha /= 2
        W2 = W - alpha * grad_W(W, B, a, X, y, n)
    return W2, alpha

def update_B(alpha, W, B, a, X, y):
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5
    b = B[:, 1]
    b = b.reshape(-1, 1)
    b2 = b - alpha * grad_B(W, B, a, X, y, n)
    B2 = np.tile(b2.reshape(-1), (n, 1)).T
    while loss_function(y, sigmoid(W.T @ X + B2, a)) > loss:
        alpha /= 2
        b2 = b - alpha * grad_B(W, B, a, X, y, n)
        B2 = np.tile(b2.reshape(-1), (n, 1)).T
    return B2, alpha


def linear_model(m, n, p, max_iterations, a, precision, X, y):

    # Initialisation random pour les poids (76,987)
    # W, b = initialization_wavier(m, p)

    # Initialisation de Xavier/Glorot pour les poids (73,513)
    # W, b = initialization_xavier(m, p)

    # Initialisation de Lecun pour les poids (77,388)
    W, b = initialization_lecun(m, p)

    B = np.tile(b, (n, 1)).T
    alpha_W = 0.1 * np.linalg.norm(W, 'fro') / np.linalg.norm(grad_W(W, B, a, X, y, n), 'fro')
    alpha_B = 0.1 * np.linalg.norm(B, 'fro') / np.linalg.norm(grad_B(W, B, a, X, y, n), 'fro')
    for _ in range(max_iterations):
        loss_prev = loss_function(y, sigmoid(W.T @ X + B, 1))
        W, alpha_W = update_W(alpha_W, W, B, a, X, y)
        B, alpha_B = update_B(alpha_B, W, B, a, X, y)
        loss = loss_function(y, sigmoid(W.T @ X + B, 1))
        print(loss)
        if abs(loss - loss_prev) < precision:
            print("break")
            break
    return W, b


if __name__ == "__main__":

    # Charger le fichier .mat
    nom_fichier_mat = '../data/data_doc.mat'
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
    m, n = Xts.shape
    p, _ = yts.shape

    num_iterations=100000
    a = 0.05
    precision = 1e-5

    W, b = linear_model(m, n, p, num_iterations, a, precision, Xts, yts)

    test(Xvr, W, b)
