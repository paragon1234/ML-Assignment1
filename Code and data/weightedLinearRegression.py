#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import inv
import math
import pandas as pd
import matplotlib.pyplot as plt

def read_data(inpFileX, inpFileY):
    X = pd.read_csv(inpFileX)
    Y = pd.read_csv(inpFileY)

    X = np.array(X)
    Z = np.ones(len(X))
    Z = Z[:, np.newaxis]
    X1 = np.concatenate((Z,X),axis = 1)
    Y = np.array(Y)
    return X1, X, Y

def calculateJ_error(X, theta, Y):
    return np.sum(np.square(X.dot(theta) - Y))/(2*len(X))

def linear_reg_normal_eqn_param(X1, Y):
    return inv(np.transpose(X1)@X1) @ np.transpose(X1) @ Y

def compute_weight_matrix(X, x0, t):
    a = [math.sqrt(math.exp(-((x - x0)**2)/(2*t*t))) for x in X]
    return np.diag(a)
    
def weighted_lin_reg_normal_eqn_param(X1, Y, W):
    X_ = W@X1
    Y_ = W@Y
    return inv(np.transpose(X_)@(X_)) @ np.transpose(X_) @ Y_

#plot Weighted Linear Regression lines at multiple points, linear regression line and dataset
def plotDataAndLinRegFit(X1, X, Y, W, wTheta):
    Y1 = np.dot(X1, W)
    plt.plot(X, Y1)
    plt.plot(X, Y, 'or')
    for i in range(len(wTheta)):
        plt.plot(X, np.dot(X1, wTheta[i]), '--g')
    plt.ylim(-2.5, 3.5)
    plt.xlim(-6, 13)
    plt.show()

#Perform weighted Linear regression for multiple points and plot
def weighted_lin_reg_plot(X1, X, Y, theta, t):
    dataPoints = [-1.0, 3.0, 6.0, 11.0]
    wTheta = np.zeros(len(dataPoints)*2).reshape((len(dataPoints),2, 1))
    for i in range(len(dataPoints)):
        W = compute_weight_matrix(X, dataPoints[i], t)
        wTheta[i] = weighted_lin_reg_normal_eqn_param(X1, Y, W)
    plotDataAndLinRegFit(X1, X, Y, theta, wTheta)


#Grab command line input
if len(sys.argv[1:]) < 3:
    print("Usage: <path_of_file_containing_x> <path_of_file_containing_y> <tau>")
    sys.exit(1)
inpFileX = sys.argv[1]
inpFileY = sys.argv[2]
t = float(sys.argv[3])

X1, X, Y = read_data(inpFileX, inpFileY) 
#t = 0.8
theta = linear_reg_normal_eqn_param(X1, Y) 
print("linear Regression Weights = ", theta)
weighted_lin_reg_plot(X1, X, Y, theta, t)

t = [0.1, 0.3, 2, 10]
for i in t:
    weighted_lin_reg_plot(X1, X, Y, theta, i)
