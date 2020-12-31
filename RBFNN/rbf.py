#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:04 2020

IMC: lab assignment 3

@author: pagutierrez
"""

# TODO Include all neccesary imports
import pickle
import click
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import pinv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time



@click.command()

# TODO Include the rest of parameters...

'''
    Parameters 
        isflag
        show_default
        default
'''

@click.option('--train_file', '-t', default=None, required=False,
              help=u'Name of the file with training data.')

@click.option('--test_file', '-T', default=None, required=False,
              help=u'Name of the file with test data.')

@click.option('--classification', '-c', is_flag=True, default=False, required=False, show_default=True,
              help=u'The problem considered is a classification problem.')

@click.option('--ratio_rbf', '-r', default=0.1, required=False, show_default=True,
              help=u'Ratio of RBF neurons (as a fraction of 1) with respect to the total number of patterns.')

@click.option('--l2', '-l', is_flag=True, default=False, required=False, show_default=True,
              help=u'Use L2 regularization instead of L1 (logistic regression).')

@click.option('--eta', '-e', default=0.01, required=False, show_default=True,
              help=u'Value of the regularization parameter for logistic regression.')

@click.option('--outputs', '-o', default=1, required=False, show_default=True,
              help=u'Number of columns that will be used as target variables (all at the end).')

@click.option('--pred', '-p', is_flag=True, default=False, show_default=True,
              help=u'Use the prediction mode.') # KAGGLE
@click.option('--model', '-m', default="", show_default=False,
              help=u'Directory name to save the models (or name of the file to load the model, if the prediction mode is active).') # KAGGLE


def train_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model, pred):
    """ 5 executions of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 5 executions with different seeds.
    """

    if not pred:    

        if train_file is None:
            print("You have not specified the training file (-t)")
            return

        # If test_file is not specified, we will use train_file as testing data
        if test_file is None:
            test_file = train_file
        
        train_mses = np.empty(5)
        train_ccrs = np.empty(5)
        test_mses = np.empty(5)
        test_ccrs = np.empty(5)
    
        for s in range(1,6,1):   
            print("-----------")
            print("Seed: %d" % s)
            print("-----------")     
            np.random.seed(s)
            train_mses[s-1], test_mses[s-1], train_ccrs[s-1], test_ccrs[s-1] = \
                train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, \
                             model and "{}/{}.pickle".format(model, s) or "")
            print("Training MSE: %f" % train_mses[s-1])
            print("Test MSE: %f" % test_mses[s-1])
            print("Training CCR: %.2f%%" % train_ccrs[s-1])
            print("Test CCR: %.2f%%" % test_ccrs[s-1])
        
        print("******************")
        print("Summary of results")
        print("******************")
        print("Training MSE: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
        print("Test MSE: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("Training CCR: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("Test CCR: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
            
    else:
        # KAGGLE
        if model is None:
            print("You have not specified the file with the model (-m).")
            return

        # Obtain the predictions for the test set
        predictions = predict(test_file, model)

        # Print the predictions in csv format
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""            
            s += str(index)
            
            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))
                
            print(s)
            


def train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ One execution of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 1 executions.

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        outputs: int
            Number of variables that will be used as outputs (all at the end
            of the matrix)
        model_file: string
            Name of the file where the model will be written

        Returns
        -------
        train_mse: float
            Mean Squared Error for training data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        test_mse: float 
            Mean Squared Error for test data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        train_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
        test_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
    """
    train_inputs, train_outputs, test_inputs, test_outputs = read_data(train_file, 
                                                                        test_file,
                                                                        outputs)

    # Measure training time
    #start_time = time.time()


    #TODO: Obtain num_rbf from ratio_rbf
    num_rbf = math.floor(train_inputs.shape[0]*ratio_rbf)
    
    print("Number of RBFs used: %d" %(num_rbf))

    kmeans, distances, centers = clustering(classification, train_inputs, 
                                              train_outputs, num_rbf)
    
    # Calculate radii and r_matrix
    radii = calculate_radii(centers, num_rbf)
    r_matrix = calculate_r_matrix(distances, radii)


    if not classification:
        coefficients = invert_matrix_regression(r_matrix, train_outputs)
    else:
        logreg = logreg_classification(r_matrix, train_outputs, l2, eta)
        
        # Print coefficients
        # print(np.sum(np.abs(logreg.coef_)>(10**(-5))))

    print("--- %s seconds ---" % (time.time() - start_time))
    """
    TODO: Obtain the distances from the centroids to the test patterns
          and obtain the R matrix for the test set
    """

    # Calculate radii and r_matrix test data
    distances_test = kmeans.transform(X=test_inputs)
    r_matrix_test = calculate_r_matrix(distances_test, radii)

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification' : classification,            
            'radii' : radii,
            'kmeans' : kmeans
        }
        if not classification:
            save_obj['coefficients'] = coefficients
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)
    
    # # # # # # # # # # #

    if not classification:
        """
        TODO: Obtain the predictions for training and test and calculate
              the MSE
        """

        # Calculate train and test outputs
        train_outputs_obtained = np.dot(r_matrix,coefficients)
        test_outputs_obtained = np.dot(r_matrix_test,coefficients)

        train_mse = mean_squared_error(train_outputs, train_outputs_obtained)
        test_mse = mean_squared_error(test_outputs, test_outputs_obtained)
        train_ccr = 0
        test_ccr = 0

    else:
        """
        TODO: Obtain the predictions for training and test and calculate
              the CCR. Obtain also the MSE, but comparing the obtained
              probabilities and the target probabilities
        """

        # Calculate CCR and train probabilities for mse
        train_ccr = logreg.score(r_matrix,train_outputs)*100
        test_ccr = logreg.score(r_matrix_test,test_outputs)*100
        
        train_probabilities_obtained = logreg.predict_proba(r_matrix)
        test_probabilities_obtained = logreg.predict_proba(r_matrix_test)

        # binary encode --> sparse = False return an array, not need to transform
        onehot_encoder = OneHotEncoder(sparse=False)
        train_outputs_encoded = onehot_encoder.fit_transform(train_outputs)
        test_outputs_encoded = onehot_encoder.fit_transform(test_outputs)

        train_mse = mean_squared_error(train_outputs_encoded,train_probabilities_obtained)
        test_mse = mean_squared_error(test_outputs_encoded,test_probabilities_obtained)


        # Test confusion matrix
        #predictions = logreg.predict(r_matrix_test)
        #fail_matrix = (predictions != np.transpose(test_outputs).ravel())
        #print(confusion_matrix(test_outputs,predictions))

        # Print index of fails in predictions
        # print(np.where(fail_matrix)[0])



    return train_mse, test_mse, train_ccr, test_ccr

def read_data(train_file, test_file, outputs):
    """ Read the input data
        It receives the name of the input data file names (training and test)
        and returns the corresponding matrices

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        outputs: int
            Number of variables to be used as outputs
            (all at the end of the matrix).
              
        Returns
        -------
        train_inputs: array, shape (n_train_patterns,n_inputs)
            Matrix containing the inputs for the training patterns
        train_outputs: array, shape (n_train_patterns,n_outputs)
            Matrix containing the outputs for the training patterns
        test_inputs: array, shape (n_test_patterns,n_inputs)
            Matrix containing the inputs for the test patterns
        test_outputs: array, shape (n_test_patterns,n_outputs)
            Matrix containing the outputs for the test patterns
    """

    #TODO: Complete the code of the function

    train_data = pd.read_csv(train_file,header=None)

    #Take all columns - number of outputs
    train_inputs = train_data.values[:,0:-outputs]

    #Take final columns (outputs)
    train_outputs = train_data.values[:,-outputs:]

    test_data = pd.read_csv(test_file,header=None)
    test_inputs = test_data.values[:,0:-outputs]
    test_outputs = test_data.values[:,-outputs:]

    return train_inputs, train_outputs, test_inputs, test_outputs



def init_centroids_classification(train_inputs, train_outputs, num_rbf):
    """ Initialize the centroids for the case of classification
        This method selects, approximately, num_rbf/num_clases
        patterns for each class.

        Parameters
        ----------
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        centroids: array, shape (num_rbf,n_inputs)
            Matrix with all the centroids already selected
    """
    
    #TODO: Complete the code of the function

    # Take stratify initial centroids
    centroids, a, b, c = train_test_split(train_inputs, train_outputs, train_size=num_rbf, stratify=train_outputs)

    return centroids

def clustering(classification, train_inputs, train_outputs, num_rbf):
    """ It applies the clustering process
        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification(). 
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        centers: array, shape (num_rbf,n_inputs)
            Centers after the clustering
    """

    #TODO: Complete the code of the function

    if classification:
        
        # Set initial centroids
        centroids = init_centroids_classification(train_inputs,train_outputs,num_rbf)

        kmeans = KMeans(n_clusters=num_rbf,init=centroids,n_init=1,max_iter=500).fit(train_inputs)

        # Obtain center and distances
        centers =  kmeans.cluster_centers_
        distances = kmeans.transform(X=train_inputs)

    else:
        
        # Set initial centroids: Randomly
        kmeans = KMeans(n_clusters=num_rbf,init='random',n_init=1,max_iter=500).fit(train_inputs)

        centers =  kmeans.cluster_centers_
        distances = kmeans.transform(X=train_inputs)

    return kmeans, distances, centers

def calculate_radii(centers, num_rbf):
    """ It obtains the value of the radii after clustering
        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Parameters
        ----------
        centers: array, shape (num_rbf,n_inputs)
            Centers from which obtain the radii
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
    """

    #TODO: Complete the code of the function

    #Calculati radii matrix
    radii = np.sum(squareform(pdist(centers,metric='euclidean')),axis=1)/(2*(num_rbf-1))

    return radii

def calculate_r_matrix(distances, radii):
    """ It obtains the R matrix
        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias
        
        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
            
        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
    """

    #TODO: Complete the code of the function

    #Calculate r_matrix as slides
    aux = -1*pow(distances,2)/(2*pow(radii,2))
    aux = np.exp(aux)

    # Add a ones column
    r_matrix = np.append(aux,np.ones([len(aux),1]),1)

    return r_matrix

def invert_matrix_regression(r_matrix, train_outputs):
    """ Inversion of the matrix for regression case
        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
              
        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value 
            of the bias 
    """

    #TODO: Complete the code of the function

    # Calculate the inverse of the matrix 
    r_matrix_transpose = r_matrix.transpose()
    r_matrix_inverse = np.dot(r_matrix_transpose,r_matrix)
    r_matrix_inverse = pinv(r_matrix_inverse)
    coefficients = np.dot(r_matrix_inverse,r_matrix_transpose)

    #Other way --> coefficients = pinv(r_matrix)

    coefficients = coefficients.dot(train_outputs)

    return coefficients

def logreg_classification(matriz_r, train_outputs, l2, eta):
    """ Performs logistic regression training for the classification case
        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
              
        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression
            Scikit-learn logistic regression model already trained
    """

    #TODO: Complete the code of the function

    regularization = 'l1'

    if l2 == True:
        regularization = 'l2'


    logreg = LogisticRegression(C=(1/eta),solver='liblinear',penalty=regularization).fit(matriz_r,train_outputs.ravel())
    
    return logreg


def predict(test_file, model_file):
    """ Performs a prediction with RBFNN model
        It obtains the predictions of a RBFNN model for a test file, using two files, one
        with the test data and one with the model

        Parameters
        ----------
        test_file: string
            Name of the test file
        model_file: string
            Name of the file containing the model data

        Returns
        -------
        test_predictions: array, shape (n_test_patterns,n_outputs)
            Predictions obtained with the model and the test file inputs
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]
    
    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    radii = saved_data['radii']
    classification = saved_data['classification']
    kmeans = saved_data['kmeans']
    
    test_distancias = kmeans.transform(test_inputs)    
    test_r = calculate_r_matrix(test_distancias, radii)    
    
    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coefficients']
        test_predictions = np.dot(test_r, coeficientes)
        
    return test_predictions
    
if __name__ == "__main__":
    train_rbf_total()
