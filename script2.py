# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:17:56 2021

@author: Erick Muñiz Morales 
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets

#Son diccionarios
boston = datasets.load_boston()

#Funciones auxiliares

def rmse(objetivo, estimaciones):
    return np.sqrt(metrics.mean_squared_error(objetivo, estimaciones)
                  )

def adjusted_r2(objetivo, estimaciones, n, k):
    r2 = metrics.r2_score(objetivo, estimaciones)
    return 1 - (1-r2)*(n-1) / (n - k - 1)

def evaluar_modelo(objetivo, estimaciones, n, k):
    return {
        "rmse": rmse(objetivo, estimaciones),
        "mae": metrics.mean_absolute_error(objetivo, estimaciones),
        "adjusted_r2": adjusted_r2(objetivo, estimaciones, n, k)
           }

modelo_ols = LinearRegression()

modelo_ols.fit(X=boston["data"], y=boston["target"])

modelo_ols_preds = modelo_ols.predict(boston["data"])


RESULTADOS = {}

N = boston["data"].shape[0]

RESULTADOS["ols"] = evaluar_modelo(
    boston["target"],
    modelo_ols_preds,
    N,
    len(modelo_ols.coef_)
)

#print(RESULTADOS)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(boston["data"], boston["target"],
                                                    test_size=0.33, random_state=13)
# Preparamos el modelo
modelo_ols = LinearRegression()

# Ajustamos el modelo entrenándolo
modelo_ols.fit(X=X_train, y=y_train)
modelo_ols_train_preds = modelo_ols.predict(X_train)

# Obtenemos los resultados del modelo recién entrenado
RESULTADOS["ols_train"] = evaluar_modelo(
    y_train,
    modelo_ols_train_preds,
    X_train.shape[0],
    len(modelo_ols.coef_)
)




