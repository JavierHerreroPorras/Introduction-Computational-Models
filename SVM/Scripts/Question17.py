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
from sklearn.metrics import plot_confusion_matrix


# Load the dataset.
data = pd.read_csv('train_spam.csv',header=None)
X_train = data.iloc[:,:-1].values
y_train = data.iloc[:,-1].values

data = pd.read_csv('test_spam.csv',header=None)
X_test = data.iloc[:,:-1].values
y_test = data.iloc[:,-1].values

# Data standarization
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# This step is to find the best parameters for the model C = 10 and gamma = 0.001
'''
Cs = np.logspace(-2, 1, num=4, base=10)
Gs = np.logspace(-3, 3, num=7, base=10)
svm_model = svm.SVC(kernel='rbf')
optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs),n_jobs=-1,cv=5)
optimo.fit(X_train,y_train)

# Obtaining the accuracy
print("Accuracy obtained: {}".format(optimo.score(X_test,y_test)))
print(optimo.best_estimator_)
'''

svm_model = svm.SVC(kernel='rbf', C = 10, gamma = 0.001)
svm_model.fit(X_train,y_train)
print("Confusion matrix obtained: ")
class_names = [ "Spam", "Not Spam"]
disp = plot_confusion_matrix(svm_model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion matrix")

plt.show()