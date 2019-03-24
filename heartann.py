# Artificial Neural Network

# Data Prepocessing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing the dataset
dataset = pd.read_csv('heart_processed.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making the ANN

# importing the Keras libraries and packages 
import keras 
from keras.models import Sequential
from keras.layers import Dense 

# Intinialising the ANN
classifier = Sequential()

# Adding the input layer and the first Hidden layer 
classifier.add(Dense(activation="relu", input_dim=13, units=7, kernel_initializer="uniform"))

# Adding the output layer 
classifier.add(Dense(activation="sigmoid", input_dim=13, units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=5000)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from keras.models import load_model

model = load_model('my_model.h5')

model.predict(X_test)

new_pred = model.predict(sc.transform(np.array([[60,1,4,130,206,0,2,132,1,2,2,2,7]])))

np.array([[67,1,4,120,229,0,2,129,1,2,2,2,7]])
