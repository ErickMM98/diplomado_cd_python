# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:44:40 2021

@author: Erick Muñiz Morales @Erickmm98
K - means
"""


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = [14, 14]
np.random.seed(42)

vehiculos = pd.read_csv("data/vehiculos_procesado_con_grupos.csv").drop(
                                                                    ["fabricante", 
                                                                     "modelo", 
                                                                     "transmision", 
                                                                     "traccion", 
                                                                     "clase", 
                                                                     "combustible", 
                                                                     "consumo"], 
                                                                    axis=1)


datos_numericos = vehiculos.select_dtypes([int, float])
print(datos_numericos.columns)
datos_categoricos = vehiculos.select_dtypes([object, "category"])

#Hay valores faltantes 
#print(vehiculos.isna().sum())

for col in datos_numericos.columns:
    datos_numericos[col].fillna(datos_numericos[col].mean(), inplace=True)

#Es importante normalizar 
from sklearn.preprocessing import MinMaxScaler

datos_numericos_normalizado = MinMaxScaler().fit_transform(datos_numericos)
datos_numericos_normalizado = pd.DataFrame(datos_numericos_normalizado,
                                               columns=datos_numericos.columns)


datos_categoricos_codificados = pd.get_dummies(datos_categoricos, 
                                               drop_first=True)


vehiculos_procesado = pd.concat([datos_numericos_normalizado, datos_categoricos_codificados], axis=1)


from sklearn.cluster import KMeans
estimador_kmedias = KMeans(random_state=42, n_clusters=8)

estimador_kmedias.fit(vehiculos_procesado)