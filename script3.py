# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:23:43 2021

@author: Erick Muñiz Morales
"""

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import datasets

#Datos de cáncer benigno y maligno
cancer_datos = datasets.load_breast_cancer()
#Tenemos dimensión fractal

#Datos
cancer_df = pd.DataFrame(cancer_datos["data"],
                           columns=cancer_datos["feature_names"]
                          )

cancer_df["objetivo"] = cancer_datos["target"]
cancer_df["objetivo"] = cancer_df["objetivo"].replace({1:0,0:1})

#print(cancer_df['objetivo'].value_counts(True))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train_df, test_df = train_test_split(cancer_df, test_size=0.4)

variables_entrenamiento = cancer_datos["feature_names"]
variable_objetivo = "objetivo"


columna_entrenamiento = "worst area"

#plt.plot(train_df[columna_entrenamiento], train_df.objetivo, '.r')
#plt.xlabel("Peor area encontrada en los nucleos de la imagen")
#plt.ylabel("Diagnostico (Maligno|Beningno)")

modelo_ols = LinearRegression()

modelo_ols.fit(train_df[[columna_entrenamiento]],
               train_df[variable_objetivo])

predicciones = modelo_ols.predict(test_df[[columna_entrenamiento]])

#print(predicciones[:10])

#plt.plot(test_df[columna_entrenamiento], test_df.objetivo, '.r')
#plt.plot(test_df[columna_entrenamiento], predicciones, '.b')
#plt.xlabel("Peor area encontrada en los nucleos de la imagen")
#plt.ylabel("Diagnostico (Maligno|Beningno)")

def funcion_logistica(x,L=1,k=1,x0=0):
    return L / (1+np.exp(-k*(x-x0)))

predicciones_probabilidades = list(map(funcion_logistica,predicciones))


#plt.plot(test_df[columna_entrenamiento], test_df.objetivo, '.r')
#plt.plot(test_df[columna_entrenamiento], predicciones, '.b')
#plt.plot(test_df[columna_entrenamiento], predicciones_probabilidades,'.g')

#plt.xlabel("Peor area encontrada en los nucleos de la imagen")
#plt.ylabel("Diagnostico (Maligno|Beningno)")


from functools import partial

#??partial

funcion_logit_k5 = partial(funcion_logistica, k=5)

predicciones_probabilidades = list(map(funcion_logit_k5, predicciones))

#plt.plot(test_df[columna_entrenamiento], test_df.objetivo, '.r')
#plt.plot(test_df[columna_entrenamiento], predicciones, '.b')
#plt.plot(test_df[columna_entrenamiento], predicciones_probabilidades,'.g')

#plt.xlabel("Peor area encontrada en los nucleos de la imagen")
#plt.ylabel("Diagnostico (Maligno|Beningno)");


from sklearn.linear_model import LogisticRegression

X = cancer_df[variables_entrenamiento]
y = cancer_df[variable_objetivo]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = LogisticRegression(solver="liblinear")
clf.fit(X_train, y_train)
predicciones = clf.predict(X_test)

predicciones[:10]

predicciones_probabilidades = clf.predict_proba(X_test)
print(predicciones)

plt.hist(predicciones_probabilidades)

probs_df = pd.DataFrame(predicciones_probabilidades)

X = X_test.reset_index().copy()
X["objetivo"] = y_test.tolist()
X["prediccion"] = predicciones
X = pd.concat([X, probs_df], axis=1)
print(X[["objetivo", "prediccion", 0, 1]].head(20))