# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 16:06:11 2021

Evaluación de modelos

@author: Erick Muñiz Morales
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets


cancer_datos = datasets.load_breast_cancer()

cancer_df = pd.DataFrame(cancer_datos["data"],
                           columns=cancer_datos["feature_names"]
                          )

cancer_df["objetivo"] = cancer_datos.target
cancer_df["objetivo"] = cancer_df["objetivo"].replace({0:1, 1:0})



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = cancer_df[cancer_datos.feature_names]
y = cancer_df["objetivo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


modelo = LogisticRegression(solver="liblinear")

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
clases_reales = y_test
predicciones_probabilidades = modelo.predict_proba(X_test)

def tupla_clase_prediccion(y_real, y_pred):
    return list(zip(y_real, y_pred))

print(tupla_clase_prediccion(clases_reales, predicciones)[:])

"""
Como nota, 
(1,0) -> Tienes cáncer pero que el modelo dice que no
(0,1) -> NO ienes cáncer pero que el modelo dice que sí

#Casos favorables
(1,1) -> Tienes cáncer y sí lo pronosticas con cáncer
(0,0) -> No tienes cáncer y no lo pronosticas con cáncer
"""

def VP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==1])

def VN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==0])
    
def FP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==1])

def FN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==0])


print("""
Verdaderos Positivos: {}
Verdaderos Negativos: {}
Falsos Positivos: {}
Falsos Negativos: {}
""".format(
    VP(clases_reales, predicciones),
    VN(clases_reales, predicciones),
    FP(clases_reales, predicciones),
    FN(clases_reales, predicciones)    
))

    
#Métricas 
# La razón de los correctos entre todos
def exactitud(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    vn = VN(clases_reales, predicciones)
    return (vp+vn) / len(clases_reales)

exactitud(clases_reales, predicciones)

from sklearn import metrics
metrics.accuracy_score(clases_reales, predicciones)

# Positivos buenos entre los buenos y los falsos
def precision(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    fp = FP(clases_reales, predicciones)
    return vp / (vp+fp)

precision(clases_reales, predicciones)

metrics.average_precision_score(clases_reales, predicciones)

# Positivos entre los positivos y los que debían ser postivios
def sensibilidad(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    fn = FN(clases_reales, predicciones)
    return vp / (vp+fn)

sensibilidad(clases_reales, predicciones)
metrics.recall_score(clases_reales, predicciones)

#Matriz de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix(clases_reales, predicciones)

#F1 nos dice que tan bueno en sensibilidad y presición
def puntuacion_f1(clases_reales, predicciones):
    precision_preds = precision(clases_reales, predicciones)
    sensibilidad_preds = sensibilidad(clases_reales, predicciones)
    return 2*(precision_preds*sensibilidad_preds)/(precision_preds+sensibilidad_preds)

puntuacion_f1(clases_reales, predicciones)

metrics.f1_score(clases_reales, predicciones)

#Radio de falsos positivos
def fpr(clases_reales, predicciones):
    return (FP(clases_reales, predicciones) / (
             FP(clases_reales, predicciones) + VN(clases_reales, predicciones)
             )
           )
fpr(clases_reales, predicciones)

"""
-------------- Humbral
"""
# Sobre el humbral 

df = pd.DataFrame({"clase_real":clases_reales,
                   "clase_pred": predicciones,
                   "probabilidades_0":modelo.predict_proba(X_test)[:,0],
                    "probabilidades_1":modelo.predict_proba(X_test)[:,1],
                  })
df["sum_probas"] = df.probabilidades_0 + df.probabilidades_1

df.query("probabilidades_1>0.5 & clase_pred==0")
df.query("probabilidades_0>0.5 & clase_pred==1")


def probabilidades_a_clases(predicciones_probabilidades, umbral=0.5):
    predicciones = np.zeros([len(predicciones_probabilidades), ])
    predicciones[predicciones_probabilidades[:,1]>=umbral] = 1
    return predicciones

from ipywidgets import widgets, fixed, interact
@interact(umbral=widgets.FloatSlider(min=0.01, max=0.99, step=0.01, value=0.01))
def evaluar_umbral(umbral):
    predicciones_en_umbral = probabilidades_a_clases(predicciones_probabilidades, umbral)
    sensibilidad_umbral = metrics.recall_score(clases_reales, predicciones_en_umbral)
    fpr_umbral = fpr(clases_reales, predicciones_en_umbral)
    precision_umbral = precision(clases_reales, predicciones_en_umbral) 
    print( """
    Precision: {:.3f}
    Sensibilidad:{:.3f}
    Ratio de Alarma: {:.3f}
    """.format(
        precision_umbral,
        sensibilidad_umbral,
        fpr_umbral
    ))
    
def evaluar_umbral(umbral):
    predicciones_en_umbral = probabilidades_a_clases(predicciones_probabilidades, umbral)
    precision_umbral = precision(clases_reales, predicciones_en_umbral)
    sensibilidad_umbral = metrics.recall_score(clases_reales, predicciones_en_umbral)
    fpr_umbral = fpr(clases_reales, predicciones_en_umbral)
    return precision_umbral, sensibilidad_umbral, fpr_umbral


rango_umbral = np.linspace(0., 1., 1000)
sensibilidad_umbrales = []
precision_umbrales = []
fpr_umbrales = []

for umbral in rango_umbral:
    precision_umbral, sensibilidad_umbral, fpr_umbral = evaluar_umbral(umbral)
    precision_umbrales.append(precision_umbral)
    sensibilidad_umbrales.append(sensibilidad_umbral)
    fpr_umbrales.append(fpr_umbral)
    
plt.plot(sensibilidad_umbrales, precision_umbrales);
plt.ylabel("Precision")
plt.xlabel("Ratio de Verdaderos positivos (sensibilidad)")
plt.title("Curva Precision-Recall");


def grafica_precision_recall(clases_reales, predicciones_probabilidades):
    precision_, recall_, _ = metrics.precision_recall_curve(
        clases_reales, predicciones_probabilidades[:,1])

    plt.step(recall_, precision_, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall_, precision_, step='post', alpha=0.2,
                 color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Curva Precision-Recall');
    plt.show()


grafica_precision_recall(clases_reales, predicciones_probabilidades)

#Area bajo de la curva AUC
plt.plot(fpr_umbrales, sensibilidad_umbrales);
plt.xlabel("Ratio de Falsos positivos (FPR)")
plt.ylabel("Ratio de Verdaderos positivos (sensibilidad)")
plt.title("Curva ROC");

#Fuente chida 
"""
https://www.datasciencecentral.com/profiles/blogs/roc-curve-explained-in-one-picture
"""

"""
COMO DECIDIR UN HUMBRAL
"""


import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cancer_datos = datasets.load_breast_cancer()

cancer_df = pd.DataFrame(cancer_datos["data"],
                           columns=cancer_datos["feature_names"]
                          )

cancer_df["objetivo"] = cancer_datos.target
cancer_df["objetivo"] = cancer_df["objetivo"].replace({0:1, 1:0})

X = cancer_df[cancer_datos.feature_names]
y = cancer_df["objetivo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression(solver="liblinear")

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
clases_reales = y_test
predicciones_probabilidades = modelo.predict_proba(X_test)

probas = modelo.predict_proba(X_test)[:5]
probas

umbral_decision = 0.5

probas[:,1]>=umbral_decision

umbral_decision = 0.1

probas[:,1]>=umbral_decision

def softmax(coste_fp, coste_fn):
    return np.exp(coste_fp) / (np.exp(coste_fn)+np.exp(coste_fp))

coste_fn = 1
coste_fp = 2
softmax(coste_fp, coste_fn)

from ipywidgets import widgets, interact

@interact
def calculo_umbral(
    coste_fp=widgets.FloatSlider(min=1, max=10, step=0.1, value=1),
    coste_fn=widgets.FloatSlider(min=1, max=10, step=0.1, value=1),
):
    return softmax(coste_fp, coste_fn)

coste_fn = 10
coste_fp = 1
umbral_decision = calculo_umbral(coste_fp, coste_fn)
print(umbral_decision)
decisiones = probabilidades_a_clases(probas, umbral_decision)
decisiones


class BusinessLogisticRegression(LogisticRegression):
        
    def decision_de_negocio(self, X, coste_fp=1, coste_fn=1, *args, **kwargs):
        probs = self.predict_proba(X)
        umbral_decision = calculo_umbral(coste_fp, coste_fn)
        print("Umbral de decision: {}".format(umbral_decision))
        decisiones = probabilidades_a_clases(probs, umbral_decision)
        return decisiones
        
modelo_negocio = BusinessLogisticRegression(solver="liblinear")

modelo_negocio.fit(X_train, y_train)


@interact(
    coste_fp=widgets.FloatSlider(min=1.,max=10.,step=.1,value=1.),
    coste_fn=widgets.FloatSlider(min=1.,max=10.,step=.1,value=1.)
)
def decision_negocio(coste_fp, coste_fn):
    predicciones = modelo_negocio.decision_de_negocio(X_test, coste_fp, coste_fn)
    print(confusion_matrix(clases_reales, predicciones))