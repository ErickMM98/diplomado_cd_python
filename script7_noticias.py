# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:19:01 2021
Práctica del clasificador de texto
@author: Erick Muñiz Morales erickmm98
"""


import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer

noticias = pd.read_csv("data/noticias.csv").iloc[:2000,]

noticias.shape

#print(noticias)

with open("data/stopwords-es.json") as fname:
    stopwords_es = json.load(fname)
    
#print(stopwords_es)

vectorizador = TfidfVectorizer(strip_accents="unicode", stop_words=stopwords_es)

vectorizador.fit_transform(noticias.descripcion)



from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from scipy.sparse import issparse


# http://rasbt.github.io/mlxtend/
class DenseTransformer(BaseEstimator):
    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

pipeline_gaussiano = make_pipeline(
    vectorizador,
    DenseTransformer(),
    GaussianNB()
)


pipeline_gaussiano.fit(X=noticias.descripcion, y=noticias.categoria)

pipeline_gaussiano.predict(noticias.descripcion)

from sklearn.metrics import f1_score


def f1_multietiqueta(estimador, X, y):
    preds = estimador.predict(X)
    return f1_score(y, preds, average="micro")

#No funciona con f1 score
#cross_val_score(pipeline_gaussiano, noticias.descripcion, noticias.categoria, scoring="f1")

def f1_multietiqueta(estimador, X, y):
    preds = estimador.predict(X)
    return f1_score(y, preds, average="micro")

cross_val_score(pipeline_gaussiano, noticias.descripcion, noticias.categoria, scoring=f1_multietiqueta,cv=10)

pipeline_gaussiano = make_pipeline(
    TfidfVectorizer(strip_accents="unicode", stop_words=stopwords_es, max_features=1000),
    DenseTransformer(),
    GaussianNB()
)