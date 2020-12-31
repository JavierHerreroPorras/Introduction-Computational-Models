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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from statistics import mean

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

# Make StratifiedKFold instance. In this case K = 5.
skf = StratifiedKFold(n_splits=5) 
lst_accuracy_kfold = []

# Numpy array. We will store the values of accuracy, C and gamma for each parameter combination
accuracy_obtained = np.empty((0,3),float) 

# Ranges for C and gamma
C = [10e-03, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3];
gamma = [10e-03, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3];


for c in C:
  for g in gamma:
    lst_accuracy_kfold = []
    for train_index in skf.split(X_train, y_train):
      # For each training process, we train the model with k-1 subsets for training and 1 subset for validation
      svm_model = svm.SVC(kernel='rbf', C = c, gamma=g)
      svm_model.fit(X_train[train_index[0]],y_train[train_index[0]])

      # We accumulate the test error and then, we obtain the mean.
      lst_accuracy_kfold.append(svm_model.score(X_train[train_index[1]],y_train[train_index[1]]))

    accuracy_obtained = np.append(accuracy_obtained,[[mean(lst_accuracy_kfold),c,g]],axis = 0)

# Obtain the combinations which have the best accuracy. We take the first one (result[0][0]).
result = np.where(accuracy_obtained[:,0] == np.amax(accuracy_obtained[:,0]))
print('Best combination of parameters (acc, C, gamma): ', accuracy_obtained[result[0][0],:])

# We take the parameters and train the model with all training patterns and obtain the final accuracy
bestC = accuracy_obtained[result[0][0],:][1]
bestGamma = accuracy_obtained[result[0][0],:][2]
svm_model = svm.SVC(kernel='rbf', C = bestC, gamma=bestGamma)
svm_model.fit(X_train,y_train)

print("Final accuracy obtained: {}".format(svm_model.score(X_test,y_test)))