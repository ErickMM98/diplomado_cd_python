# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:40:44 2021

@author: mumoe
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = [12, 12]
np.random.seed(42)

pelis = pd.read_csv("data/datos_peliculas.csv")
pelis.shape

pelis = pelis.drop("pelicula", axis=1)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score

variable_objetivo_clasificacion = "genero"
variables_independientes_clasificacion = pelis.drop(
    variable_objetivo_clasificacion, axis=1).columns

X_train, X_test, y_train, y_test = train_test_split(
    pelis[variables_independientes_clasificacion],
    pelis[variable_objetivo_clasificacion], test_size=0.20)


k_categorias = len(y_train.unique())
k_categorias


clasificador_knn = KNeighborsClassifier(n_neighbors=10, 
                                        weights="uniform")

clasificador_knn.fit(X_train, y_train)

preds = clasificador_knn.predict(X_test)
f1_score(y_test, preds, average="micro")
#Un f1 bastante bajo xd 0.3617021276595745
print(f1_score(y_test, preds, average="micro"))


clasificador_knn = KNeighborsClassifier(n_neighbors=10, 
                                        weights="distance")

clasificador_knn.fit(X_train, y_train)

preds = clasificador_knn.predict(X_test)
print(f1_score(y_test, preds, average="micro"))
