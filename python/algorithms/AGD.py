from python.algorithms.functions import *
import numpy as np


def AGD_function(m, n, p, max_iterations, a, stop_condition, X, y, init, accel, evol_beta, validation=False, X_val=[],
                 y_val=[], lbd=1e-10, alpha_b=0.5, tau=0.05):
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
    alpha_W = tau * np.linalg.norm(W, 'fro') / np.linalg.norm(grad_W(W, B, a, X, y), 'fro')
    alpha_B = tau * np.linalg.norm(b) / np.linalg.norm(grad_B(W, B, a, X, y), 'fro')

    loss_vect = []
    val_vect = []
    best_valid = 100
    best_W = W
    best_b = b
    if accel == "polyak":
        W_prev = W
        B_prev = B
        for k in range(1, max_iterations + 1):

            if evol_beta == "const":
                beta = 0.5

            if evol_beta == "paul":
                beta = evol_beta_paul(k)

            if evol_beta == "nesterov":
                beta, alpha_b = evol_beta_nesterov(alpha_b)

            loss_prev = loss_function_L2(y, sigmoid(W.T @ X + B, a), lbd, W)
            W, alpha_W, W_prev = update_W_Polyak(alpha_W, W, B, a, X, y, W_prev, beta, lbd)
            B, alpha_B, B_prev = update_B_Polyak(alpha_B, W, B, a, X, y, B_prev, beta)
            loss = loss_function_L2(y, sigmoid(W.T @ X + B, a), lbd, W)
            if validation:
                b_val = B[:, 1]
                b_val = b_val.reshape(-1, 1)
                _, n = X_val.shape
                B_val = np.tile(b_val.reshape(-1), (n, 1)).T
                loss_val = loss_function_L2(y_val, sigmoid(W.T @ X_val + B_val, a), lbd, W)
                val_vect.append(loss_val)
                if best_valid > loss_val:
                    best_valid = loss_val
                    best_W = W
                    best_b = B[:, 1]
            print(loss)
            loss_vect.append(loss)
            if abs(loss - loss_prev) < stop_condition:
                print("break")
                break
        if validation:
            W = best_W
            b = best_b
        return W, b, loss_vect, val_vect


    elif accel == "nesterov":
        W_prev = W
        B_prev = B
        for k in range(1, max_iterations + 1):

            if evol_beta == "const":
                beta = 0.5

            if evol_beta == "paul":
                beta = evol_beta_paul(k)

            if evol_beta == "nesterov":
                beta, alpha_b = evol_beta_nesterov(alpha_b)

            loss_prev = loss_function_L2(y, sigmoid(W.T @ X + B, a), lbd, W)
            W, alpha_W, W_prev = update_W_Nesterov(alpha_W, W, B, a, X, y, W_prev, beta, lbd)
            B, alpha_B, B_prev = update_B_Nesterov(alpha_B, W, B, a, X, y, B_prev, beta)
            loss = loss_function_L2(y, sigmoid(W.T @ X + B, a), lbd, W)
            if validation:
                b_val = B[:, 1]
                b_val = b_val.reshape(-1, 1)
                _, n = X_val.shape
                B_val = np.tile(b_val.reshape(-1), (n, 1)).T
                loss_val = loss_function_L2(y_val, sigmoid(W.T @ X_val + B_val, a), lbd, W)
                val_vect.append(loss_val)
                if best_valid > loss_val:
                    best_valid = loss_val
                    best_W = W
                    best_b = B[:, 1]

            print(loss)
            loss_vect.append(loss)
            if abs(loss - loss_prev) < stop_condition:
                print("break")
                break
        if validation:
            W = best_W
            b = best_b
        return W, b, loss_vect, val_vect


def update_W_Polyak(alpha, W, B, a, X, y, W_prev, beta, lbd):
    loss = loss_function_L2(y, sigmoid(W.T @ X + B, a), lbd, W)
    alpha *= 1.5

    # Mise à jour des poids en utilisant la méthode de Polyak's
    W2 = W - alpha * grad_W_L2(W, B, a, X, y, lbd) + beta * (W - W_prev)

    if loss_function_L2(y, sigmoid(W2.T @ X + B, a), lbd, W2) > loss:
        alpha /= 2
        W2 = W - alpha * grad_W_L2(W, B, a, X, y, lbd)

    return W2, alpha, W


def update_B_Polyak(alpha, W, B, a, X, y, B_prev, beta):
    _, n = X.shape
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5

    b = B[:, 1]
    b = b.reshape(-1, 1)
    b_prev = B_prev[:, 1]
    b_prev = b_prev.reshape(-1, 1)

    # Mise à jour des poids en utilisant la méthode de Polyak's
    b2 = b - alpha * grad_B(W, B, a, X, y) + beta * (b - b_prev)
    B2 = np.tile(b2.reshape(-1), (n, 1)).T

    if loss_function(y, sigmoid(W.T @ X + B2, a)) > loss:
        alpha /= 2
        b2 = b - alpha * grad_B(W, B, a, X, y)
        B2 = np.tile(b2.reshape(-1), (n, 1)).T

    return B2, alpha, B


def update_W_Nesterov(alpha, W, B, a, X, y, W_prev, beta, lbd):
    loss = loss_function_L2(y, sigmoid(W.T @ X + B, a), lbd, W)
    alpha *= 1.5

    # Mise à jour des poids en utilisant la méthode de Nesterov
    W2 = W - alpha * grad_W_L2(W - beta * (W - W_prev), B, a, X, y, lbd) + beta * (W - W_prev)

    while loss_function_L2(y, sigmoid(W2.T @ X + B, a), lbd, W2) > loss:
        alpha /= 2
        W2 = W - alpha * grad_W_L2(W, B, a, X, y, lbd)

    return W2, alpha, W


def update_B_Nesterov(alpha, W, B, a, X, y, B_prev, beta):
    _, n = X.shape
    loss = loss_function(y, sigmoid(W.T @ X + B, a))
    alpha *= 1.5

    b = B[:, 1]
    b = b.reshape(-1, 1)
    b_prev = B_prev[:, 1]
    b_prev = b_prev.reshape(-1, 1)

    # Mise à jour des poids en utilisant la méthode de Nesterov
    b2 = b - alpha * grad_B(W, B - beta * (B - B_prev), a, X, y) + beta * (b - b_prev)
    B2 = np.tile(b2.reshape(-1), (n, 1)).T

    while loss_function(y, sigmoid(W.T @ X + B2, a)) > loss:
        alpha /= 2
        b2 = b - alpha * grad_B(W, B, a, X, y)
        B2 = np.tile(b2.reshape(-1), (n, 1)).T

    return B2, alpha, B
