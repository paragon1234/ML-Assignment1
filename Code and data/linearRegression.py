#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib.colors import LogNorm

def normalizeData(X1):
    mu = np.mean(X1)
    std = np.std(X1)
    X1 = (X1 - mu)/std
    return X1

#X is the input data and X1 is input data appended with a column of one's
def read_and_normalize_data(inpFileX, inpFileY):
    X = pd.read_csv(inpFileX)
    Y = pd.read_csv(inpFileY)

    X = np.array(X)
    X = normalizeData(X)
    Z = np.ones(len(X))
    Z = Z[:, np.newaxis]
    X1 = np.concatenate((Z,X),axis = 1) #added a column of 1 to X
    Y = np.array(Y)
    return X, X1, Y

#compute Gradient Descent Error
def calculateJ(X1, W, Y):
    J = 0
    for i in range(len(X)):
        J = J + (Y[i] - np.dot(X1[i], W))**2
    return J/(2*len(X1))

#Calculate Gradient Descent gradient
def calculateGradJ(X1, W, Y, j):
    J = 0
    for i in range(len(X)):
        J = J + (Y[i] - np.dot(X1[i], W))*X1[i][j]
    return J/len(X1)
   
#Gradient Descent Algorithm 
def Gradient_descent(X1, X, Y, learning_rate, J_diff):
    W = np.array([0.0, 0.0])
    prev_J = 0.0

    W_arr = np.array([]).reshape(0,2) #stores weight at each iteration of GD
    J_arr = np.array([]) #stores error at each iteration of GD
    iter = 0

    while 1:
        iter += 1
        J = calculateJ(X1, W, Y)
        #print("error in this iter=",J)
        if abs(prev_J - J) < J_diff:
            break
        tempW = np.copy(W) #Creates tempW and copy in it elements of W
        W_arr = np.vstack([W_arr, W])
        J_arr = np.append(J_arr, J)
        for i in range(len(W)):
            tempW[i] = W[i] + learning_rate*calculateGradJ(X1, W, Y, i)
        W[:] = tempW
        prev_J = J
    return W, iter, W_arr, J_arr

def plotDataAndLinRegLine(X1, Y, W):
    Y1 = np.dot(X1, W)
    plt.plot(X, Y1) #plot the line
    plt.plot(X, Y, 'or') #plot the data
    plt.show()

#Grab command line input
if len(sys.argv[1:]) < 4:
    print("Usage: <path_of_file_containing_x> <path_of_file_containing_y> <learning rate> <time_gap_in_seconds>")
    sys.exit(1)
inpFileX = sys.argv[1]
inpFileY = sys.argv[2]
learning_rate = float(sys.argv[3])
time_gap = float(sys.argv[4])

#Data pre-Processing
X, X1, Y = read_and_normalize_data(inpFileX, inpFileY) 
#learning_rate = 0.2
J_diff = 1e-11 #0.0000012 #stopping criteria for difference between error
W, iter, W_arr, J_arr = Gradient_descent(X1, X, Y, learning_rate, J_diff)

print("-------------------")
print("final Weights = ", W)
print("No. of iterations = ", iter)
print("Learning Rate = ", learning_rate)
print("stopping criteria is difference in error in subsequent iteration less than ", J_diff)
print("-------------------")

#Plot the dataset and the Linear Regression Line
plotDataAndLinRegLine(X1, Y, W)

#3D mesh with animation of error value
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(0, 2, 0.01)
y = np.arange(-0.02, 0.02, 0.001)
x,y = np.meshgrid(x,y)
z = np.array([calculateJ(X1, np.array([w0, w1]), Y) #calcJ(w0, w1) 
               for w0, w1 in zip(np.ravel(x), np.ravel(y))])
z = z.reshape(x.shape)
ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', alpha=0.5)
ax.view_init(20,75)

line, = ax.plot([], [], [], 'r', lw=2)
point, = ax.plot([], [], [], 'ro')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point,

def animate(i, line, point, W_arr, J_arr):
    line.set_data(W_arr[:i,0], W_arr[:i,1])
    line.set_3d_properties(J_arr[:i])
    point.set_data(W_arr[i-1:i,0], W_arr[i-1:i,1])
    point.set_3d_properties(J_arr[i-1:i])
    return line, point,

anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(line, point, W_arr, J_arr),
                               frames=J_arr.shape[0], interval=time_gap*1000, 
                               repeat_delay=5, blit=True)
plt.show()

##Contour Plot with Animation
def contour_plot_and_animation(W_arr, J_arr):
    fig,ax = plt.subplots(figsize=(10,6))
    ax.contour(x, y, z, levels = 80, norm=LogNorm(), cmap=plt.cm.jet)
    line, = ax.plot([], [], 'r', lw=2)
    point, = ax.plot([], [], 'ro')
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point,
    
    def animate(i, line, point, W_arr):
        line.set_data(W_arr[:i,0], W_arr[:i,1])
        point.set_data(W_arr[i-1:i,0], W_arr[i-1:i,1])
        return line, point,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(line, point, W_arr),
                                   frames=J_arr.shape[0], interval=time_gap*1000,
                                   repeat_delay=5, blit=True)
    plt.show()

contour_plot_and_animation(W_arr, J_arr)

#Contour plot for different learning_rate
learning_rate = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
for n in learning_rate:
    W, iter, W_arr, J_arr = Gradient_descent(X1, X, Y, n, J_diff)
    contour_plot_and_animation(W_arr, J_arr)

