# -*- Chantal Simó-*-
"""Bayesian Optimization.ipynb
Original file is located at
    https://colab.research.google.com/drive/1UnuVh3-b9U3gajErwC54j4hEM4Ds5wiV
"""

!pip install bayesian-optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
from bayes_opt import BayesianOptimization, UtilityFunction
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
LABELS = ['NO','YES']
# --------------------------------------------------------------------------------
# Prepare the data.
df = pd.read_csv(r'/content/framingham.csv')
df = df.dropna()

y = df['TenYearCHD']
X = df.drop('TenYearCHD', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            stratify = y,
                                        random_state = 42)

os =  RandomOverSampler()
X_train_res, y_train_res = os.fit_resample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Define the black box function to optimize.
def black_box_function(C):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C)
    model.fit(X_train_scaled, y_train_res)
    y_score = model.decision_function(X_test_scaled) #probabilidad de clasificación .4 - 40%
    f = roc_auc_score(y_test, y_score)
    return f
# Set range of C to optimize for.
# bayes_opt requires this to be a dictionary.
pbounds = {"C": [0.1, 10]}
# Create a BayesianOptimization optimizer,
# and optimize the given black_box_function.
optimizer = BayesianOptimization(f = black_box_function,
                                 pbounds = pbounds,
                                 random_state = 4)

optimizer.maximize(init_points = 5, n_iter = 10)

print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

# Define the black box function to optimize.
def black_box_function(C):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C)
    model.fit(X_train_scaled, y_train_res)
    y_score = model.decision_function(X_test_scaled) #probabilidad de clasificación .4 - 40%
    f = roc_auc_score(y_test, y_score)
    return f
# Set range of C to optimize for.
# bayes_opt requires this to be a dictionary.
pbounds = {"C": [0.1, 10]}
# Create a BayesianOptimization optimizer,
# and optimize the given black_box_function.
optimizer = BayesianOptimization(f = black_box_function,
                                 pbounds = pbounds,
                                 random_state = 4)

optimizer.maximize(init_points = 5, n_iter = 10)

print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

model1 = SVC(C=0.8243855804208278)
model1.fit(X_train_scaled, y_train_res)
y_pred = model1.predict(X_test_scaled)

#definimos funciona para mostrar los resultados
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))

mostrar_resultados(y_test,y_pred)

"""---
* Función que te arroje matriz de confusión
"""

def matriz_conf(ax,bx):
    a = list(ax)
    b = list(bx)
    matrix=np.zeros((2,2))
    for i in range(len(a)):
        #1=p, 0=n
        if int(a[i])==1 and int(b[i])==0:
            matrix[0,0]+1 #True Positives
        elif int(a[i])==-1 and int(b[i])==1:
            matrix[0,1]+1 #False Positives
        elif int(a[i])==0 and int(b[i])==1:
            matrix[1,0]+1 #False Negatives
        elif int(a[i])==0 and int(b[i])==0:
            matrix[1,1]+1 #True Negatives
    return matrix
matriz_conf(y_pred,y_test)

# FUNCION VISTA EN CLASE
def matriz confusion(a,b):
    count_0_0 = 0
    count_0_1 = 0
    count_1_0 = 0
    count_1_1 = 0
    for i in range(len(a)):
      if a[i] == b[i]:
        if a[i] == 0 and b[i] == 0
          count_0_0 =  count_0_0 + 1
        if a[i] == 1 and b[i] == 1
          count_1_1 =  count_1_1 + 1
      if a[i] !== b[i]:
        if a[i] == 0 and b[i] == 1
          count_0_1 =  count_0_1 + 1
        if a[i] == 1 and b[i] == 0
          count_1_0 =  count_1_0 + 1

print("TP: " + str(count_0_0),"FT: " + str(count_0_1),"TF: " + str(count_1_0),"TN: " + str(count_1_1), )