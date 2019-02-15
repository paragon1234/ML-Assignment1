#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from pandas.compat import StringIO
import matplotlib.pyplot as plt

#read data from dat files
#X0 contains X-data for y=0 ie alaska.
#X1 contains X-data for y=1 ie canada
def read_data(inpFileX, inpFileY):
    with open(inpFileY) as fn:
        content = fn.readlines()
    Y = [i.strip()=='Alaska' for i in content]
    Y = np.array(Y) 
    #phi = np.sum(Y)/Y.size #numpy.count_nonzero(Y)

    X = np.loadtxt(inpFileX)
    X1 = [X[i] for (i,j) in enumerate(Y) if j==True]
    X1 = np.array(X1)
    X0 = [X[i] for (i,j) in enumerate(Y) if j==False]
    X0 = np.array(X0)
    return X0, X1, content

def get_mean_and_variance(X0, X1):
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    sigma0 = np.cov(X0, rowvar=False, bias = True)
    sigma1 = np.cov(X1, rowvar=False, bias = True)
    #covariance calculated over the whole dataset
    sigma = np.cov(np.concatenate((X0, X1)), rowvar=False, bias=True)
    
    #Covariance can also be calculated as weighted average of positive and negative dataset
    return mu0, mu1, sigma, (X0.shape[0]*sigma0 + X1.shape[0]*sigma1)/(X0.shape[0] + X1.shape[0]), sigma0, sigma1
    
#Create 2D meshGrid for plotting
def create_mesh_grid():
    x1 = np.linspace(50, 175, 251)
    x2 = np.linspace(0, 550, 551)
    return np.meshgrid(x1, x2)

#plot the data and linear as well as quadratic decision boundary
def plot_data(X, Y, theta0, theta1, mu0, mu1, quad):
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
    ax.legend()

    #plt.plot(X0[:,0], X0[:,1], 'bs', X1[:,0], X1[:,1], 'rs')
    #plt.legend()

    #Plot the linear decision boundary
    #Y1 = (theta0_mean - np.dot(X[:,0], theta1_mean[0]))/theta1_mean[1]   
    #plt.plot(X[:,0], Y1, '--g')
    Y0 = (theta0 - np.dot(X[:,0], theta1[0]))/theta1[1]  
    plt.plot(X[:,0], Y0, '--r')

    #Plot quadratic decision boundary
    x1, x2 = create_mesh_grid()
    plt.contour(x1, x2, quad, levels=[0], colors='green')
    #plt.contour(x1, x2, lin, levels=[0], colors='green')

    #Plot the mean values
    plt.plot(mu0[0], mu0[1], 'xy')
    plt.plot(mu1[0], mu1[1], 'xy')

    plt.show()

#Compute the values of the decision boundary at the meshgrid
def GDA_decision_boundary(x, mu0, mu1, sigma0, sigma1, sigmaInv0, sigmaInv1, size0, size1):
    const_term1 = np.log(size0/size1) - 0.5*(np.transpose(mu0) @ sigmaInv0 @ mu0) + 0.5 * (np.transpose(mu1) @ sigmaInv1 @ mu1)
    const_term2 = (np.log((np.linalg.det(sigma1)) / (np.linalg.det(sigma0)))) * 0.5
    lin_term = np.transpose(x) @ sigmaInv0 @ mu0 - np.transpose(x) @ sigmaInv1 @ mu1
    quad_term = -0.5 * (np.transpose(x) @ sigmaInv0 @ x) + 0.5 * (np.transpose(x) @ sigmaInv1 @ x)
    return const_term1 + const_term2 + lin_term + quad_term

#Compoute the contour of the decision boundary
def compute_GDA_decision_boundary_contour(mu0, mu1, sigma0, sigma1, size0, size1):
    x1, x2 = create_mesh_grid()
    z = np.zeros(x1.shape)

    sigmaInv0 = np.linalg.inv(sigma0)
    sigmaInv1 = np.linalg.inv(sigma1)
    for i in range(0, len(z)):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]]).reshape(2, 1)
            z[i][j] = GDA_decision_boundary(x, mu0, mu1, sigma0, sigma1, sigmaInv0, sigmaInv1, size0, size1)

    return z

#Grab command line input
if len(sys.argv[1:]) < 2:
    print("Usage: <path_of_file_containing_x> <path_of_file_containing_y>")
    sys.exit(1)

inpFileX = sys.argv[1]
inpFileY = sys.argv[2]

#read the data
X0, X1, Y = read_data(inpFileX, inpFileY)
mu0, mu1, sigma, sigma_mean, sigma0, sigma1 = get_mean_and_variance(X0, X1)

#Compute Linear decision boundary parameters
sigmaInv = np.linalg.inv(sigma)
#theta0 = (-0.5) * (np.transpose(mu0-mu1) @ sigmaInv @ (mu0-mu1)) + np.log(len(X0)/len(X1))
theta0 = 0.5 * (np.transpose(mu0) @ sigmaInv @ mu0) - 0.5 * (np.transpose(mu1) @ sigmaInv @ mu1) + np.log(len(X1)/len(X0))
theta1 = sigmaInv @ (mu0-mu1)

#sigmaInv_mean = np.linalg.inv(sigma_mean)
##theta0_mean = (-0.5) * (np.transpose(mu0-mu1) @ sigmaInv_mean @ (mu0-mu1)) + np.log(len(X0)/len(X1))
#theta0_mean = 0.5 * (np.transpose(mu0) @ sigmaInv_mean @ mu0) - 0.5 * (np.transpose(mu1) @ sigmaInv @ mu1) + np.log(len(X1)/len(X0))
#theta1_mean = sigmaInv_mean @ mu0 - sigmaInv_mean @ mu1
#print("covariance as weighted average of canada and alaska data =", sigma_mean)
#plot_data(np.concatenate((X0, X1)), Y, theta0, theta1, theta0_mean, theta1_mean, mu0, mu1)

#Print the values
print("mean Alaska = ", mu1)
print("mean Canada =", mu0) 
print("covariance over whole dataset = ", sigma)
print("covariance of Alaska = ", sigma1)
print("covariance of Canada = ", sigma0)

#compute Quadratic decision boundary
#lin = compute_GDA_decision_boundary_contour(mu0, mu1, sigma,  sigma,  len(X0), len(X1)) 
quad = compute_GDA_decision_boundary_contour(mu0, mu1, sigma0, sigma1, len(X0), len(X1)) 
plot_data(np.concatenate((X0, X1)), Y, theta0, theta1, mu0, mu1, quad)
