from python.algorithms.functions import *
import random
def SGD_function(m, n, p, a, stop_condition, X, y, init, epoch):

    if init == "random":
        # Initialisation random pour les poids (76,987)
        W, b = initialization_random(m, p)
    elif init == "xavier":
        # Initialisation de Xavier/Glorot pour les poids (73,513)
        W, b = initialization_xavier(m, p)
    elif init == "lecun":
        # Initialisation de Lecun pour les poids (77,388)
        W, b = initialization_lecun(m, p)

    alpha = 0.1
    b = b.reshape(-1, 1)
    B = np.tile(b.reshape(-1), (n, 1)).T
    loss_vect = []
    for e in range(epoch):
        idx_list = list(range(0, n))
        for k in range(0, n):
            loss_prev = loss_function(y, sigmoid(W.T @ X + B, a))
            # Pick a random element from the list
            idx = random.choice(idx_list)
            idx_list.remove(idx)
            # Update W
            W = W - alpha * grad_W(W, b, a, X[:,idx], y[:,idx])
            # Update B
            b = b - alpha * grad_B(W, b, a, X[:,idx], y[:,idx])

            B = np.tile(b.reshape(-1), (n, 1)).T

            loss = loss_function(y, sigmoid(W.T @ X + B, a))
            if abs(loss - loss_prev) < stop_condition:
                print("break")
                break
        loss_vect.append(loss)
        print(loss)
    return W, b, loss_vect