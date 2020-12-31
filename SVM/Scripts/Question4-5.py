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

from sklearn import svm

# I get values of C and gamma by command line.
print("C = {}".format(float(sys.argv[1])));
print("gamma = {}".format(float(sys.argv[2])));

# Load the dataset. Reading input variables and saving them in X. Also we
# read prediction labels and save them in y.
data = pd.read_csv('dataset2.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Train the SVM model.
# For question 4, change the kernel to 'linear'
svm_model = svm.SVC(kernel='rbf', C = float(sys.argv[1]), gamma = float(sys.argv[2]))
svm_model.fit(X, y)

# Plot the points
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Plot the support vectors class regions, the separating hyperplane and the margins
plt.axis('tight')
# |->Plot support vectors
#plt.scatter(svm_model.support_vectors_[:,0], svm_model.support_vectors_[:,1], 
            #marker='+', s=100, zorder=10, cmap=plt.cm.Paired)
# |-> Extract the limits
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()
# |-> Create a grid with all the points and then obtain the SVM 
#    score for all the points
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])
# |-> Plot the results in a countour
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], 
            linestyles=['--', '-', '--'], levels=[-1, 0, 1])

plt.xlabel('x1')
plt.ylabel('x2')

plt.show()