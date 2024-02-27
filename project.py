import scipy.io

# Charger le fichier .mat
nom_fichier_mat = 'C:/Users/Cedri/Downloads/FirstOMFL-SO/Projet/docum-classi/data_doc.mat'
donnees = scipy.io.loadmat(nom_fichier_mat)

# Afficher les clés (noms des variables) présentes dans le fichier .mat
print("Noms des variables dans le fichier :", donnees.keys())

# Accéder aux variables en utilisant les noms corrects
if 'Xts' in donnees:
    Xts = donnees['Xts']
    print("Variable Xts lue avec succès :", Xts)
else:
    print("La variable 'Xts' n'a pas été trouvée dans le fichier.")

if 'yts' in donnees:
    yts = donnees['yts']
    print("Variable yts lue avec succès :", yts)
else:
    print("La variable 'yts' n'a pas été trouvée dans le fichier.")

if 'Xvr' in donnees:
    Xvr = donnees['Xvr']
    print("Variable Xvr lue avec succès :", Xvr)
else:
    print("La variable 'Xvr' n'a pas été trouvée dans le fichier.")