from python.algorithms.functions import *
from python.algorithms.SGD import SGD_function
from python.algorithms.GD import linear_model
from python.algorithms.AGD import AGD_function
from python.tests.GDvsSGD import comparisonGD_SGD
from python.tests.GDvsAGD import comparisonGD_AGD
from python.tests.validation_test import validation_test
from python.tests.sigmoid_test import sigmoid_test
import numpy as np

if __name__ == "__main__":
    # Charger le fichier .mat
    matfile = '../data/data_doc.mat'
    Xts, yts, Xvr = load_mat_file(matfile)

    yts = one_hot_encoding(yts)
    Xts, yts = shuffle(Xts, yts)
    m, n = Xts.shape
    p, _ = yts.shape

    # Paramètre a de la sigmoid
    a = 0.05

    # Choisir initialisation : random, xavier, lecun
    initialisation = "random"
    # Choisir acceleration : none, polyak, nesterov
    acceleration = "nesterov"
    # Choisir evolution beta dans le cas d'AGD : const, paul , nesterov
    evol_beta = "nesterov"

    # Comparaison de SGD et GD
    # comparisonGD_SGD(m, n, p, a, Xts, yts,epoch = 100)
    # Comparaison de notres GD et nos AGD
    # comparisonGD_AGD(m, n, p, a, Xts, yts,epoch = 300)
    validation_test(a, Xts, yts, Xvr)
    # Meilleure valeur pour notre sigmoid
    # sigmoid_test(Xts, yts)
