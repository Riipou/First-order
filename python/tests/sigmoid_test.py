from python.algorithms.functions import *
from python.algorithms.GD import linear_model
from python.algorithms.AGD import AGD_function
import numpy as np
import random

def sigmoid_test(Xts, yts, epoch = 100):
    X_train, y_train, X_val, y_val = split_data(Xts, yts, split_value=0.95)
    m, n = X_train.shape
    p, _ = y_train.shape
    stop_condition = 1e-10
    for a in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,5.0,10.0]:
        random.seed(42)
        acceleration = "nesterov"
        initialisation = "random"
        evol_beta = "nesterov"
        file_name = f"../results/sigmoid/a={a}.txt"
        with open(file_name, 'w', newline='') as file:
            W, b, loss_vect, val_vect = AGD_function(m, n, p, epoch, a, stop_condition, X_train, y_train, initialisation, acceleration, evol_beta, validation=True, X_val=X_val, y_val=y_val)
            i = 0
            for l in loss_vect:
                i += 1
                file.write(f"({i},{l})\n")
            file.write(f"\nValidation:\n")
            i = 0
            for l in val_vect:
                i += 1
                file.write(f"({i},{l})\n")
        #test(Xvr, W, b, epoch, a, stop_condition, initialisation, acceleration, evol_beta)

        pass