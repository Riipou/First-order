import time
from SGD import SGD_function
from GD import linear_model
import random
def comparisonGD_SGD(m, n, p, a, Xts, yts, epoch):
    stop_condition = 1e-10
    start_time = time.time()
    random.seed(42)
    W, b = linear_model(m, n, p, epoch, a, stop_condition, Xts, yts, "random", "none", "const")
    end_time = time.time()-start_time
    print("Temps d'éxecution pour GD avec "+str(epoch)+" epoch = "+str(end_time)+" secondes")
    random.seed(42)
    W, b = SGD_function(m, n, p, a, stop_condition, Xts, yts, "random", epoch)
    print("Temps d'éxecution pour SGD avec " + str(epoch) + " = " + str(end_time) + " secondes")
    end_time = time.time()-end_time
    pass