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


def update_W_Polyak(alpha, W, B, a, X, y, W_prev, beta):
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5

    # Mise à jour des poids en utilisant la méthode de Polyak's
    W2 = W - alpha * grad_W(W, B, a, X, y, n) + beta * (W - W_prev)

    if loss_function(y, sigmoid(W2.T @ X + B, a)) > loss:
        alpha /= 2
        W2 = W - alpha * grad_W(W, B, a, X, y, n)

    return W2, alpha, W


def update_B_Polyak(alpha, W, B, a, X, y, B_prev, beta):
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5

    b = B[:, 1]
    b = b.reshape(-1, 1)
    b_prev = B_prev[:, 1]
    b_prev = b_prev.reshape(-1, 1)

    # Mise à jour des poids en utilisant la méthode de Polyak's
    b2 = b - alpha * grad_B(W, B, a, X, y, n) + beta * (b - b_prev)
    B2 = np.tile(b2.reshape(-1), (n, 1)).T

    if loss_function(y, sigmoid(W.T @ X + B2, a)) > loss:
        alpha /= 2
        b2 = b - alpha * grad_B(W, B, a, X, y, n)
        B2 = np.tile(b2.reshape(-1), (n, 1)).T

    return B2, alpha, B


def update_W_Nesterov(alpha, W, B, a, X, y, W_prev, beta):
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5

    # Mise à jour des poids en utilisant la méthode de Nesterov
    W2 = W - alpha * grad_W(W - beta * (W - W_prev), B, a, X, y, n) + beta * (W - W_prev)

    while loss_function(y, sigmoid(W2.T @ X + B, a)) > loss:
        alpha /= 2
        W2 = W - alpha * grad_W(W, B, a, X, y, n)

    return W2, alpha, W


def update_B_Nesterov(alpha, W, B, a, X, y, B_prev, beta):
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5

    b = B[:, 1]
    b = b.reshape(-1, 1)
    b_prev = B_prev[:, 1]
    b_prev = b_prev.reshape(-1, 1)

    # Mise à jour des poids en utilisant la méthode de Nesterov
    b2 = b - alpha * grad_B(W, B - beta * (B - B_prev), a, X, y, n) + beta * (b - b_prev)
    B2 = np.tile(b2.reshape(-1), (n, 1)).T

    while loss_function(y, sigmoid(W.T @ X + B2, a)) > loss:
        alpha /= 2
        b2 = b - alpha * grad_B(W, B, a, X, y, n)
        B2 = np.tile(b2.reshape(-1), (n, 1)).T

    return B2, alpha, B


def linear_model(m, n, p, max_iterations, a, stop_condition, X, y, init, accel, evol_beta):
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
    alpha_B = 0.1 * np.linalg.norm(b) / np.linalg.norm(grad_B(W, B, a, X, y, n), 'fro')

    if accel == "none":
        file_name = f"../results/train/GD_{max_iterations}_ite_sigmoid_{a}_stop_{stop_condition}_{init}.txt"
        with open(file_name, 'w', newline='') as file:
            for k in range(max_iterations):
                loss_prev = loss_function(y, sigmoid(W.T @ X + B, a))
                W, alpha_W = update_W(alpha_W, W, B, a, X, y)
                B, alpha_B = update_B(alpha_B, W, B, a, X, y)
                loss = loss_function(y, sigmoid(W.T @ X + B, a))
                file.write(f"({k+1},{loss})\n")
                print(loss)
                if abs(loss - loss_prev) < stop_condition:
                    print("break")
                    break
            return W, b

    elif accel == "polyak":
        W_prev = W
        B_prev = B
        alpha_b = 0.5
        file_name = f"../results/train/AGD_polyak_{max_iterations}_ite_sigmoid_{a}_stop_{stop_condition}_{init}_{accel}_beta_{evol_beta}.txt"
        with open(file_name, 'w', newline='') as file:
            for k in range(1, max_iterations + 1):

                if evol_beta == "const":
                    beta = 0.5

                if evol_beta == "paul":
                    beta = evol_beta_paul(k)

                if evol_beta == "nesterov":
                    beta, alpha_b = evol_beta_nesterov(alpha_b)

                loss_prev = loss_function(y, sigmoid(W.T @ X + B, a))
                W, alpha_W, W_prev = update_W_Polyak(alpha_W, W, B, a, X, y, W_prev, beta)
                B, alpha_B, B_prev = update_B_Polyak(alpha_B, W, B, a, X, y, B_prev, beta)
                loss = loss_function(y, sigmoid(W.T @ X + B, a))
                print(loss)
                file.write(f"({k},{loss})\n")
                if abs(loss - loss_prev) < stop_condition:
                    print("break")
                    break
            return W, b


    elif accel == "nesterov":
        W_prev = W
        B_prev = B
        alpha_b = 0.5
        file_name = f"../results/train/AGD_nesterov_{max_iterations}_ite_sigmoid_{a}_stop_{stop_condition}_{init}_{accel}_beta_{evol_beta}.txt"
        with open(file_name, 'w', newline='') as file:
            for k in range(1, max_iterations + 1):

                if evol_beta == "const":
                    beta = 0.5

                if evol_beta == "paul":
                    beta = evol_beta_paul(k)

                if evol_beta == "nesterov":
                    beta, alpha_b = evol_beta_nesterov(alpha_b)

                loss_prev = loss_function(y, sigmoid(W.T @ X + B, a))
                W, alpha_W, W_prev = update_W_Nesterov(alpha_W, W, B, a, X, y, W_prev, beta)
                B, alpha_B, B_prev = update_B_Nesterov(alpha_B, W, B, a, X, y, B_prev, beta)
                loss = loss_function(y, sigmoid(W.T @ X + B, a))
                print(loss)
                file.write(f"({k},{loss})\n")
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
    num_iterations = 10000
    # Paramètre a de la sigmoid
    a = 0.05
    # Condition d'arrêt : différence d'erreur entre 2 epochs
    stop_condition = 1e-5

    # Choisir initialisation : random, xavier, lecun
    initialisation = "random"
    # Choisir acceleration : none, polyak, nesterov
    acceleration = "nesterov"
    # Choisir evolution beta : const, paul , nesterov
    evol_beta = "nesterov"

    W, b = linear_model(m, n, p, num_iterations, a, stop_condition, Xts, yts, initialisation, acceleration, evol_beta)

    test(Xvr, W, b, num_iterations, a, stop_condition, initialisation, acceleration, evol_beta)
