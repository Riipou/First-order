from functions import *
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import pandas as pd
from scipy.sparse import csr_matrix

if __name__ == "__main__":

    # Charger le fichier .mat
    nom_fichier_mat = 'data_doc.mat'
    donnees = loadmat(nom_fichier_mat)

    # Xts combien de fois le mot dans le texte
    if 'Xts' in donnees:
        Xts = donnees['Xts']
    else:
        print("La variable 'Xts' n'a pas été trouvée dans le fichier.")

    # id texte avec classe
    if 'yts' in donnees:
        yts = donnees['yts']
    else:
        print("La variable 'yts' n'a pas été trouvée dans le fichier.")

    # Xvr combien de fois le mot dans le texte
    if 'Xvr' in donnees:
        Xvr = donnees['Xvr']
    else:
        print("La variable 'Xvr' n'a pas été trouvée dans le fichier.")

    yts = one_hot_encoding(yts)
    Xts, yts = shuffle(Xts, yts)
    X_train, y_train, X_test, y_test = split_data(Xts, yts)
