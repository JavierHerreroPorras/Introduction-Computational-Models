#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import svm, model_selection, preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load the dataset.
data = pd.read_csv('dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Divide dataset into train part and test part.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Data standarization
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code taken from the lab assignment instructions
Cs = np.logspace(-3, 3, num=7, base=10)
Gs = np.logspace(-3, 3, num=7, base=10)
svm_model = svm.SVC(kernel='rbf')
optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs),n_jobs=-1,cv=5)
optimo.fit(X_train,y_train)

# Obtaining the accuracy and the best parameters for the model
print("Accuracy obtained: {}".format(optimo.score(X_test,y_test)))
print(optimo.best_estimator_)
print(optimo.get_params())
