# -*- coding: utf-8 -*-
"""
Status: COMPLETE

Created on Wed Sep  19 19:00:00 2019

@author: Jon Berezowski

FYS - 3002: Pattern Recognition
Week 36, Exercise Set 5 - Exercise 4.2

Using the computer, generate four two-dimensional Gaussian random sequences
with covariance matrices

0.01 0.0
0.0 0.01

and mean values [0,0], [1,1], [0,1], [1,0] 

The first two form class w1, and the other two class w2. Produce 100 vectors
from each distribution. Use the batch mode backpropagation algorithm of
Section 4.6 to train a two-layer perceptron with two hidden neurons and one
in the output. Let the activation function be the logistic one with a = 1. Plot
the error curve as a function of iteration steps. Experiment yourselves with
various values of the learning parameter mu. Once the algorithm has converged,
produce 50 more vectors from each distribution and try to classify them using
the weights you have obtained. What is the percentage classification error?
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Set up Mean Vectors and Covariance Matrix
u1 = np.array([0,0])
u2 = np.array([1,1])
u3 = np.array([0,1])
u4 = np.array([1,0])
sig = np.matrix([[0.01, 0.0], [0.0, 0.01]])
N = 100
S = 15000 #Max Iterations

#Produce sample vectors and assign to classes
x1 = np.random.multivariate_normal(u1, sig, N)
x2 = np.random.multivariate_normal(u2, sig, N)
x3 = np.random.multivariate_normal(u3, sig, N)
x4 = np.random.multivariate_normal(u4, sig, N)

x = np.concatenate([x1, x2, x3, x4])
x = np.append(x, (np.reshape(np.ones(4 * N), (-1, 1))), axis = 1)
y_true = np.atleast_2d(np.concatenate([np.zeros(2*N), np.ones(2*N)])).T

#Global Variables
w0 = np.random.rand(3,2)
w1 = np.random.rand(3,1)
a = 1
y1 = np.zeros((400,2))
y_hat = np.zeros((400,2))
mu = 0.015
error_v = []

def forward_pass(x, w0 = w0, w1 = w1, a = a, y1 = y1, y_hat = y_hat):
    #Layer 1 - two neurons
    y1 = activate(np.dot(x, w0), a) #Compute "potential" vector
    y1 = np.append(y1, (np.reshape(np.ones(len(x[:,0])), (-1, 1))), axis = 1)

    #Layer 2 - one neuron
    y_hat = activate(np.dot(y1, w1), a)
    return y_hat, y1

def backward_pass(x,y_true, a = a, w0 = w0, w1 = w1, mu = mu, S = S):
#For all layers in reverse order
    for i in range(S):
        y_hat, y1 = forward_pass(x, w0, w1)
        if i % 10 == 0:
            error_v.append(error(y_hat, y_true))
            
        #dw1 calculations
        v1 = np.dot(w1.T, x.T)
        dfdv1 = a*(activate(v1,a)*(1-activate(v1,a))).T
        y_hat = classify_y(y_hat, x, a)
        y_error = np.subtract(y_hat,y_true)
        del21 = np.multiply(dfdv1,y_error)
        dw1 = -mu*(np.dot(y1.T,del21))
        
        #dw0 calculations
        v0 = np.dot(w0.T, x.T)
        dfdv0 = a*(activate(v0,a)*(1-activate(v0,a))).T
        error_11 = np.multiply(del21, w1[0])
        error_12 = np.multiply(del21, w1[1])
        del11 = np.multiply(error_11, np.atleast_2d(dfdv0[:,0]).T)
        del12 = np.multiply(error_12, np.atleast_2d(dfdv0[:,1]).T)
        
        dw01 = np.dot(x.T, del11)
        dw02 = np.dot(x.T, del12)

        dw0 = -mu * np.concatenate([dw01, dw02], axis = 1)

        #Update Weights
        w1 = w1 + dw1
        w0 = w0 + dw0
        
        #Check Continue
        y_check, y1_results = forward_pass(x, w0, w1)
        y_results = classify_y(y_check, x, a)
        test_err = class_error(y_results, y_true)
        test_acc = (len(x[:,0]) - test_err)/len(x[:,0])
        
        if test_acc > 0.999:
            break
        

    return w1, w0, y_hat, y_error


#Activation Function: Sigmoid
def activate(v, a):
    f = (1 / (1 + np.exp(-a*v)))
    return f
"""
#Activation Function: ReLu
def activate(v, a):
    f = max(0,v)
    return f
"""

#Calculate Error
def error(y_hat, y_true = y_true):
    j_sum = 0.5*np.sum(np.square((np.subtract(y_hat, y_true))))
    return j_sum

#Classifier
def classify_y(y, x, a):
    y_results = np.zeros((len(x[:,0]),1))
    for i in range(len(y)):
        if y[i] > 0.5:
            y_results[i] = 1
        else:
            y_results[i] = 0
            
    return y_results

#Classification Error
def class_error(y_results, y_true = y_true):
    error_sum = 0
    for i in range(len(y_results)):
        if y_results[i] == y_true[i]:
            continue
        else:
            error_sum += 1

    return error_sum

#Run Test and Achieve Weights
w1_test, w0_test, y_hattest, y_errortest = backward_pass(x,y_true)

#Calculate Results
y_check, y1_results = forward_pass(x, w0_test, w1_test)
y_results = classify_y(y_check, x, a)
test_err = class_error(y_results, y_true)
test_acc = (len(x[:,0]) - test_err)/len(x[:,0])
print("Test Accuracy: " + str(test_acc))
print("Minimum Error Count: " + str(min(error_v)))

#Plot Training Data
colors = ['green', 'blue']
fig = plt.figure(figsize=(13,13))
plt.scatter(x[:,0],x[:,1], c=y_results[:,0], cmap=matplotlib.colors.ListedColormap(colors), marker='x', linewidths=3)
plt.show()

#Plot Error Curve
fig2 = plt.figure(figsize=(13,13))
x_grid = np.linspace(0, len(error_v), len(error_v))
plt.plot(x_grid, error_v)
plt.show()

#Produce 50 Additional Vectors and Test Weights
NT = 50
xt1 = np.random.multivariate_normal(u1, sig, NT)
xt2 = np.random.multivariate_normal(u2, sig, NT)
xt3 = np.random.multivariate_normal(u3, sig, NT)
xt4 = np.random.multivariate_normal(u4, sig, NT)

xt = np.concatenate([xt1, xt2, xt3, xt4])
xt = np.append(xt, (np.reshape(np.ones(4 * NT), (-1, 1))), axis = 1)
yt = np.atleast_2d(np.concatenate([np.zeros(2*NT), np.ones(2*NT)])).T

yt_check, y1_results = forward_pass(xt, w0_test, w1_test)
yt_results = classify_y(yt_check, xt, a)
test_errt = class_error(yt_results, yt)
test_acct = (len(xt[:,0]) - test_errt)/len(xt[:,0])
print("New Data Test Accuracy: " + str(test_acct))
print("Test Error Count: " + str(test_errt))

#Plot Test Data
fig = plt.figure(figsize=(13,13))
plt.scatter(xt[:,0],xt[:,1], c=yt_results[:,0], cmap=matplotlib.colors.ListedColormap(colors), marker='x', linewidths=3)
plt.show()
