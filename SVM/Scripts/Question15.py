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

# Load vocab
vocab = pd.read_csv('vocab.csv',header=None)

# Load the dataset.
data = pd.read_csv('train_spam.csv',header=None)
X_train = data.iloc[:,:-1].values
y_train = data.iloc[:,-1].values

data = pd.read_csv('test_spam.csv',header=None)
X_test = data.iloc[:,:-1].values
y_test = data.iloc[:,-1].values

# Make a copy of test (I will standarize it)
X_test_copy = X_test
y_test_copy = y_test

# Data standarization
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = svm.SVC(kernel='linear', C = 0.01, gamma = 0.001)
svm_model.fit(X_train,y_train)

# Obtaining the accuracy and the confusion matrix
print("Accuracy obtained: {}".format(svm_model.score(X_test,y_test)))
print("Confusion matrix obtained: ")
class_names = [ "Spam", "Not Spam"]
disp = plot_confusion_matrix(svm_model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion matrix")

plt.show()

predicted = svm_model.predict(X_test)

# Show which pattern was incorret classified.
for cont,pred,lab in zip(range(0,X_test.shape[0]),predicted,y_test):
  print("\n --->Emails wrong classified as Not Spam<---")
  if pred != lab and pred == 1:
      print('\n',cont)
      # Compression list, we select the words that appear in the email
      inputVariables = [vocab[0][element] for element in range(len(X_test_copy[cont])) if X_test_copy[cont][element] == 1]
      print(inputVariables)

  print("\n --->Emails wrong classified as Spam<---")
  if pred != lab and pred == 0:
      print('\n',cont)
      inputVariables = [vocab[0][element] for element in range(len(X_test_copy[cont])) if X_test_copy[cont][element] == 1]
      print(inputVariables)
