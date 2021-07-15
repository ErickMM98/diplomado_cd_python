# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:13:12 2021

@author: Erick Muñiz Morales 
Procesamiendo
"""

path_data = 'data/datos_procesamiento.csv'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datos = pd.read_csv(path_data)
datos.head()

from sklearn import preprocessing

var_numericas_df = datos.select_dtypes([int, float])
print(var_numericas_df)
var_numericas_df["col_outliers"] = datos["col_outliers"]
var_numericas_df.columns

from sklearn.impute import SimpleImputer
imputador = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean")

var_numericas_imputadas = imputador.fit_transform(var_numericas_df)

var_numericas_imputadas_df = pd.DataFrame(var_numericas_imputadas,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)


#Estandarización
"""
Sirve para el caso de restar la media y dividir por la desviación estándar
"""
escalador = preprocessing.StandardScaler()
var_numericas_imputadas_escalado_standard = escalador.fit_transform(var_numericas_imputadas)

print(var_numericas_imputadas_escalado_standard.mean(axis=0),
      var_numericas_imputadas_escalado_standard.std(axis=0))

"""
Una estandarización más robusta; usando la mediana y el rango intercuantil
"""

escalador_robusto = preprocessing.RobustScaler()
var_numericas_imputadas_escalado_robusto = escalador_robusto.fit_transform(
                                                        var_numericas_imputadas)

print(var_numericas_imputadas_escalado_robusto.mean(axis=0))


