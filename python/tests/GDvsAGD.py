from python.algorithms.functions import *
from python.algorithms.GD import linear_model
from python.algorithms.AGD import AGD_function
import numpy as np
import time
import random

def comparisonGD_AGD(m, n, p, a, Xts, yts, epoch):
    stop_condition = 1e-10
    start_time = time.time()
    random.seed(42)
    W, b, _ = linear_model(m, n, p, epoch, a, stop_condition, Xts, yts, "random")
    end_time = time.time()-start_time
    print("Temps d'éxecution pour GD avec "+str(epoch)+" epoch = "+str(end_time)+" secondes")
    random.seed(42)
    W, b, _, _= AGD_function(m, n, p, epoch, a, stop_condition, Xts, yts, "random", "nesterov", "const")
    end_time = time.time()-end_time
    print("Temps d'éxecution pour AGD nesterov avec " + str(epoch) + " = " + str(end_time) + " secondes")
    random.seed(42)
    W, b, _, _ = AGD_function(m, n, p, epoch, a, stop_condition, Xts, yts, "random", "polyak", "const")
    end_time = time.time() - end_time
    print("Temps d'éxecution pour AGD polyak avec " + str(epoch) + " = " + str(end_time) + " secondes")
    pass