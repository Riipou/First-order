from scipy.io import loadmat
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Charger le fichier .mat
nom_fichier_mat = 'data_doc.mat'
donnees = loadmat(nom_fichier_mat)

# Afficher les clés (noms des variables) présentes dans le fichier .mat
# print("Noms des variables dans le fichier :", donnees.keys())

# Xts combien de fois le mot dans le texte
if 'Xts' in donnees:
    Xts = donnees['Xts']
    # print("Variable Xts lue avec succès :\n", Xts)
else:
    print("La variable 'Xts' n'a pas été trouvée dans le fichier.")
# id texte avec classe
if 'yts' in donnees:
    yts = donnees['yts']
    # print("Variable yts lue avec succès :\n", yts)
else:
    print("La variable 'yts' n'a pas été trouvée dans le fichier.")
# Xvr combien de fois le mot dans le texte
if 'Xvr' in donnees:
    Xvr = donnees['Xvr']
    # print("Variable Xvr lue avec succès :\n", Xvr)
else:
    print("La variable 'Xvr' n'a pas été trouvée dans le fichier.")
print(Xts.shape," and ",yts.shape)

# Split data into train and test sets
Xts = Xts.T
X_train, X_test, y_train, y_test = train_test_split(Xts, yts, test_size=0.2, random_state=42)

# Define the perceptron model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=X_train.shape[1], activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary crossentropy for binary classification
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)