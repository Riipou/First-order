from python.algorithms.functions import *
from python.algorithms.SGD import SGD_function
from python.algorithms.GD import linear_model
from python.algorithms.AGD import AGD_function
from python.tests.GDvsSGD import comparisonGD_SGD
import numpy as np

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
    a = 0.2
    # Condition d'arrêt : différence d'erreur entre 2 epochs
    stop_condition = 1e-5

    # Choisir initialisation : random, xavier, lecun
    initialisation = "random"
    # Choisir acceleration : none, polyak, nesterov
    acceleration = "nesterov"
    # Choisir evolution beta dans le cas d'AGD : const, paul , nesterov
    evol_beta = "nesterov"

    file_name = f"../results/train/{acceleration}_{num_iterations}_ite_sigmoid_{a}_stop_{stop_condition}_{initialisation}_{acceleration}_beta_{evol_beta}.txt"
    with open(file_name, 'w', newline='') as file:
        if acceleration == "none":
            W, b, loss_vect = linear_model(m, n, p, num_iterations, a, stop_condition, Xts, yts, initialisation)
        elif acceleration == "nesterov" or acceleration == "polyak":
            W, b, loss_vect, _ = AGD_function(m, n, p, num_iterations, a, stop_condition, Xts, yts, initialisation, acceleration, evol_beta)

        i = 0
        for l in loss_vect:
            i+=1
            file.write(f"({i},{l})\n")

    test(Xvr, W, b, num_iterations, a, stop_condition, initialisation, acceleration, evol_beta)
