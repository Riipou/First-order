from functions import *
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


def linear_model(m, n, p, max_iterations, a, stop_condition, X, y, init):

    if init == "random":
        # Initialisation random pour les poids (76,987)
        W, b = initialization_random(m, p)
    elif init == "xavier":
        # Initialisation de Xavier/Glorot pour les poids (73,513)
        W, b = initialization_xavier(m, p)
    elif init == "lecun":
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
        if abs(loss - loss_prev) < stop_condition:
            print("break")
            break
    return W, b


if __name__ == "__main__":

    # Charger le fichier .mat
    matfile = '../data/data_doc.mat'
    Xts, yts, Xvr = load_mat_file(matfile)

    yts = one_hot_encoding(yts)
    Xts, yts = shuffle(Xts, yts)
    m, n = Xts.shape
    p, _ = yts.shape

    # Définition des variables
    num_iterations = 100000
    a = 0.05
    stop_condition = 1e-5
    # Choisir initialisation : random, xavier, lecun
    initialisation = "lecun"

    with open("../results/param_test.txt", "w") as file:
        # Garder une trace des paramètres
        file.write(f"num_iterations = {num_iterations}\n")
        file.write(f"a = {a}\n")
        file.write(f"critère d'arrêt = {stop_condition}\n")
        file.write(f"initialisation : {initialisation}\n")

    W, b = linear_model(m, n, p, num_iterations, a, stop_condition, Xts, yts, initialisation)

    test(Xvr, W, b)
