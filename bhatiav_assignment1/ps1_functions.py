"""
Deep Learning Theory and Applications, Problem Set 1
"""
# helpful libraries
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors # used to identify nearest neighbors, but not to classify
# for problem 5
import torch
import torch.nn as nn
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality


def problem2_evaluate_function_on_random_noise(N, sigma):
    """Sample N points uniformly from the interval [-1,3],
    add random noise, and output the function y = x^2 - 3x + 1

    Parameters
    ----------
    N : int
        The number of points
    sigma : float
        The standard deviation of noise to add to the randomly generated points.

    Returns
    -------
    x, y (list, list) 
        x, the randomly generated points with added noise
        y, the function evaluated at these points.
    """

    x = np.random.uniform(low=-1.0, high=3+1e-7,size = N)
    def getY(x):
        return x**2 - 3*x + 1
    noise = np.random.normal(loc = 0.0, scale = sigma, size = N)
    y = np.apply_along_axis(getY,-1,x)
    y+= noise
    
    return list(x), list(y)

def problem2_fit_polynomial(x, y, degree, regularization = None):
    """Returns optimal coefficients for a polynomial of the given degree
    to fit the data, using the Moore-Penrose Pseudoinverse (specified in the assignment)
    Note: this function only needs to function for degrees 1,2, and 9 --
    but you are welcome build something that works for any degree.
    By incorporating the value of the regularization parameter, this function should work 
    for both 2.2 and 2.3

    Parameters
    ----------
    x : list of floats 
        The input x values
    y : list of floats
        The input y values
    degree : int
        The degree of the polynomial to fit
    regularization : float
        The parameter lambda which specifies the degree of regularization to apply. Default 0.

    Returns
    -------
    list of floats  
        The coefficients of the polynomial.
    """
    
    y = np.array(y)
    y = y.reshape(-1,1)
    x = np.array(x)
    deg_ct = degree+1
    lambda_ = 0
    if(regularization):
        lambda_ = regularization
   
    x_len = len(x)
    arr_ = np.empty((deg_ct,x_len))

    for i in range(deg_ct):
        #print(i)
        arr_[i] = x**i
    arr = np.transpose(arr_)
    X = np.array(arr)
    coeffs = np.dot(np.linalg.inv(np.dot(X.transpose(),X) + np.identity(deg_ct)*lambda_),np.dot(X.transpose(),y))
    coeffs = np.concatenate(coeffs).ravel()
    return list(coeffs)

def problem3_knn_classifier(train_data, train_labels, test_data, k):
    """A kth Nearest Neighbor classified. Accepts points and training labels, 
    and returns predicted labels for each point in the dataset.

    Parameters
    ----------
    train_data : ndarray
        The training points, in an n x d array, where n is the number of points and d is the dimension.
    train_labels : list of classes
        The training labels. They should correspond directly to the points in the training data array.
    test_data : ndarray
        The unlabelled data, to be labelled by the classifier
    k : positive int
        The number of nearest neighbors to consult.

    Returns
    -------
    predicted_labels : list
        The labels outputted by the classifier for each of the test datapoints.
    """

    nn = NearestNeighbors(n_neighbors = k,algorithm = 'auto')
    nn.fit(train_data)
    distances,indices = nn.kneighbors(test_data)

    predicted_labels = []
    for i in range(len(indices)):
        labels = []
        for idx in indices[i]:
            labels.append(int(train_labels[idx]))
       
        predicted_labels.append(max(labels,key = labels.count))
        

    return predicted_labels