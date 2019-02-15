#!/usr/bin/env python

import sys
import numpy as np
from numpy.linalg import inv
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#Read the data from csv files
#X is the input data
#X1 is input data appended with a column of 1
def read_data(inpFileX, inpFileY):
    X = pd.read_csv(inpFileX)
    Y = pd.read_csv(inpFileY)

    X = np.array(X)
    Z = np.ones(len(X))
    Z = Z[:, np.newaxis]
    X1 = np.concatenate((Z,X),axis = 1) #added a column of 1 to X
    Y = np.array(Y).reshape((len(Y)))
    return X, X1, Y

def calculate_error(hTheta, Y):
    return np.sum(np.square(hTheta - Y))

def calculate_grad(X, hTheta, Y):
    diff = Y-hTheta
    term0 = np.sum(diff)
    term1 = np.sum(diff * X[:,0])
    term2 = np.sum(diff * X[:,1]) 
    return np.array([term0, term1, term2]) #.reshape((3,1))

def calculate_hessian(X, hTheta):
    h = -hTheta*(1-hTheta)
    term0 = np.sum(h)
    term1 = term3 = np.sum(h * X[:,0])
    term2 = term6 = np.sum(h * X[:,1])
    term4 = np.sum(h * X[:,0] * X[:,0])
    term5 = term7 = np.sum(h * X[:,0] * X[:,1])
    term8 = np.sum(h * X[:,1] * X[:,1])
    return np.array([term0, term1, term2, term3, term4, term5, term6, term7, term8]).reshape((3,3))

def newton_method(X, X1, Y): 
    iter = 0
    theta = np.zeros(X1.shape[1])
    hTheta = np.array([0.5 for x in X])
    prev_error = calculate_error(hTheta, Y)
    while(1):
        iter += 1
        grad = calculate_grad(X, hTheta, Y)
        hess = calculate_hessian(X, hTheta)
        theta = theta - np.linalg.inv(hess)@grad
        hTheta = 1/(1+np.exp(-X1@theta))
        error = calculate_error(hTheta, Y)
        print("error=", error)
        if abs(error - prev_error) < 1e-8:
            return theta, iter
        else:
            prev_error = error
    
#plot Logistic Regression lines at multiple points, linear regression line and dataset
def plot_data_and_decision_boundary(x, y, theta):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot and decision boundary')

    xone = x[:, 0]
    xtwo = x[:, 1]

    xone_yis1 = []
    xtwo_yis1 = []

    xone_yis0 = []
    xtwo_yis0 = []

    for x1, x2, y in zip(xone, xtwo, y):
        if y == 1:
            xone_yis1.append(x1)
            xtwo_yis1.append(x2)
        else:
            xone_yis0.append(x1)
            xtwo_yis0.append(x2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim((0, 10))
    plt.ylim((0, 10))

    y0 = plt.scatter(xone_yis0, xtwo_yis0, marker='o')
    y1 = plt.scatter(xone_yis1, xtwo_yis1, marker='x')

    x1 = np.linspace(0, 10, 21)
    x2 = [-(theta[0] + theta[1]*x)/theta[2] for x in x1]
    plt.plot(x1, x2, '--r')
    redline = mlines.Line2D([], [], color='red')
    ax.legend([y0, y1, redline], ['y=0', 'y=1', 'Logistic Boundary'])

    # fig.savefig("Plots/gda.png")
    plt.show()

#Grab command line input
if len(sys.argv[1:]) < 2:
    print("Usage: <path_of_file_containing_x> <path_of_file_containing_y>")
    sys.exit(1)

inpFileX = sys.argv[1]
inpFileY = sys.argv[2]

X, X1, Y = read_data(inpFileX, inpFileY)
theta, iter = newton_method(X, X1, Y)
print("theta=", theta)
print("iteration=", iter)
plot_data_and_decision_boundary(X,Y, theta)
