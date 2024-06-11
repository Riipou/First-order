from functions import *
from SGD import SGD_function
from GD import linear_model
from GDvsSGD import comparisonGD_SGD
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
    a = 0.05
    # Condition d'arrêt : différence d'erreur entre 2 epochs
    stop_condition = 1e-5

    # Choisir initialisation : random, xavier, lecun
    initialisation = "random"
    # Choisir acceleration : none, polyak, nesterov
    acceleration = "none"
    # Choisir evolution beta : const, paul , nesterov
    evol_beta = "nesterov"

    # Comparaison de SGD et GD
    comparisonGD_SGD(m,n,p,a,Xts,yts,epoch =100)
    # W, b = SGD_function(m, n, p, a, stop_condition, Xts, yts, initialisation, 2)
    # W, b = linear_model(m, n, p, num_iterations, a, stop_condition, Xts, yts, initialisation, acceleration, evol_beta)

    # test(Xvr, W, b, num_iterations, a, stop_condition, initialisation, acceleration, evol_beta)
