from python.algorithms.functions import *
from python.algorithms.GD import linear_model
from python.algorithms.AGD import AGD_function
import numpy as np
import random


def validation_test(a, Xts, yts, Xvr, epoch=500):
    X_train, y_train, X_val, y_val = split_data(Xts, yts, split_value=0.95)
    m, n = X_train.shape
    p, _ = y_train.shape
    stop_condition = 1e-8
    acc = "nesterov"
    initialisation = "random"
    random.seed(42)
    # acceleration = "nesterov"
    for tau in [0.05]:
        evol_beta = "nesterov"
        file_name = f"../results/initialisation_alpha/{acc}_tau={tau}.txt"
        with open(file_name, 'w', newline='') as file:
            W, b, loss_vect, val_vect = AGD_function(m, n, p, epoch, a, stop_condition, X_train, y_train,
                                                     initialisation, acc, evol_beta, validation=True, X_val=X_val,
                                                     y_val=y_val,
                                                     tau=tau)
            i = 0
            for l in loss_vect:
                i += 1
                file.write(f"({i},{l})\n")
            i = 0
            best = 1
            for l in val_vect:
                i += 1
                file.write(f"({i},{l})\n")
                if l < best:
                    best = l
            file.write(f"\nbest val error :{best}\n")
        test(Xvr, W, b, epoch, a, stop_condition, initialisation, acc, evol_beta)
        pass
