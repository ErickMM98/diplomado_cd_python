# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:11:29 2021

@author: Erick Muñiz Morales
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, feature_extraction

datos = pd.read_csv("datos_pipelines.csv")
datos.head()
