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

print("C = {}".format(float(sys.argv[1])));
print("gamma = {}".format(float(sys.argv[2])));


# Load the dataset.
data = pd.read_csv('dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Divide dataset into train part and test part with a 25% of patterns in test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Data standarization
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model.
svm_model = svm.SVC(kernel='rbf', C = float(sys.argv[1]), gamma = float(sys.argv[2]))
svm_model.fit(X_train, y_train)

# Obtaining the accuracy
print("Accuracy obtained: {}".format(svm_model.score(X_test,y_test)))
