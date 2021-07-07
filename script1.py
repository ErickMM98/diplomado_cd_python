# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:06:08 2021
Diplomado en Ciencia de Datos con Python
@author: Erick Muñiz Morales @ErickMM98
"""

import pandas as pd 
import numpy as np
import os


data = pd.read_csv("peliculas.csv")
dataset = data.select_dtypes(np.number)

objetivo = dataset['ventas']
variables_independientes = dataset[['presupuesto',
                                    'popularidad',
                                    'duracion',
                                    'puntuacion',
                                    'n_votos']]

from sklearn.linear_model import LinearRegression
modelo = LinearRegression()

modelo.fit(X=variables_independientes, y = objetivo)

modelo.intercept_
modelo.coef_

dataset['Co_predict'] = modelo.predict(variables_independientes)

#Errores

from sklearn import metrics
metrics.mean_absolute_error( y_true = dataset['ventas'], y_pred = dataset['Co_predict'])
np.sqrt(metrics.mean_absolute_error( y_true = dataset['ventas'], y_pred = dataset['Co_predict']))