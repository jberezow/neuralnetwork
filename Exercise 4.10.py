# -*- coding: utf-8 -*-
"""
Status: COMPLETE 
*Converges regularly with >90% accuracy.
Next steps for Improvement:
*Next step: RELU Activation
*Next step: Adaptive Momentum

Created on Mon Oct  07 09:33:00 2019

@author: Jon Berezowski

FYS - 3002: Pattern Recognition
Week 36, Exercise Set 5 - Exercise 4.10

Develop a program to repeat the simulation example of Section 4.10.

This section demonstrates the capability of a multilayer perceptron to classify
nonlinearly separable classes. The classification task consists of two distinct classes,
each being the union of four regions in the two-dimensional space. Each region
consists of normally distributed random vectors with statistically independent 
components and each with variance of 0.08. The mean values are different for each 
of the regions. Specifically, the regions of class 1 are formed around the mean vectors:
[0.4, 0.9]T , [2.0, 1.8]T , [2.3, 2.3]T , [2.6, 1.8]T

and those of class 2 around the values:
[1.5, 1.0]T , [1.9, 1.0]T , [1.5, 3.0]T , [3.3, 2.6]T

A total of 400 training vectors were generated, 50 from each distribution. A multilayer
perceptron, with three neurons in the first and two neurons in the second hidden
layer, were used with a single output neuron. The activation function was the logistic
one with a = 1 and the desired outputs 1 and 0, respectively, for the two classes.

Two different algorithms were used for the training, namely, the momentum and
the adaptive momentum. After some experimentation the algorithmic parameters
employed were (a) for the momentum mu = 0.05, alpha = 0.85 and (b) for the adaptive
momentum mu =  0.01, alpha = 0.85, ri = 1.05, c = 1.05, rd = 0.7. The weights were
initialized by a uniform pseudorandom distribution between 0 and 1.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Hidden or output layers for neural network
class Layer:
 
    def __init__(self, n_input, n_neurons, weights = None, bias = None):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """
 
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
        self.last_delw = np.zeros((1,n_neurons))

    def activate(self, x):
        r = np.dot(x, self.weights) - self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation
 
    def _apply_activation(self, r):
        return 1 / (1 + np.exp(-r))

    def apply_activation_derivative(self, r):
        return r * (1 - r)
 
#Neural Network Object
class NeuralNetwork:
 
    def __init__(self):
        self._layers = []
 
    def add_layer(self, layer):
        self._layers.append(layer)
 
    def feed_forward(self, x): 
        for layer in self._layers:
            x = layer.activate(x)

        return x
 
    def predict(self, x): 
        ff = self.feed_forward(x)
        y_results = np.zeros((len(x[:,0]),1))
        for i in range(len(ff)):
            if ff[i] > 0.5:
                y_results[i] = 1
            else:
                y_results[i] = 0
        """
        # One row
        if ff.ndim == 1:
            return np.argmax(ff)
 
        # Multiple rows
        return np.argmax(ff, axis=1)
        """
        return y_results
 
    def backpropagation(self, x, y, learning_rate, momentum):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """
 
        # Feed forward for the output
        output = self.feed_forward(x)
 
        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
 
            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
 
        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or x itself (for the first hidden layer)
            input_to_use = np.atleast_2d(x if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += (layer.delta * input_to_use.T * learning_rate) - (momentum * layer.last_delw)
            layer.last_delw = (layer.delta * input_to_use.T * learning_rate) - (momentum * layer.last_delw)
 
    def train(self, x, y, learning_rate, momentum, max_epochs):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """
 
        mses = []
        y_guess = []

        for i in range(max_epochs):
            for j in range(len(x)):
                self.backpropagation(x[j], y[j], learning_rate, momentum)
            if i % 10 == 0:
                mse = np.mean(np.square(y - self.feed_forward(x)))
                mses.append(mse)
                #print('Epoch: #%s, MSE: %f' % (i, float(mse)))
            y_guess.append(self.accuracy(self.predict(x), y))
            if y_guess[i] > 0.95:
                break

            if len(y_guess) > 20001:
                if y_guess[i] == y_guess[-5000] == y_guess[-20000]:
                    break
            
        return mses, y_guess
 
    @staticmethod
    def accuracy(y_pred, y_true):
        return (y_pred == y_true).mean()

#Set up Mean Vectors and Covariance Matrix
u1 = np.array([0.4, 0.9])
u2 = np.array([2.0, 1.8])
u3 = np.array([2.3, 2.3])
u4 = np.array([2.6, 1.8])
u5 = np.array([1.5, 1.0])
u6 = np.array([1.9, 1.0])
u7 = np.array([1.5, 3.0])
u8 = np.array([3.3, 2.6])
sig = np.matrix([[0.006, 0.0], [0.0, 0.006]])
N = 30

#Produce sample vectors and assign to classes
x1 = np.random.multivariate_normal(u1, sig, N)
x2 = np.random.multivariate_normal(u2, sig, N)
x3 = np.random.multivariate_normal(u3, sig, N)
x4 = np.random.multivariate_normal(u4, sig, N)
x5 = np.random.multivariate_normal(u5, sig, N)
x6 = np.random.multivariate_normal(u6, sig, N)
x7 = np.random.multivariate_normal(u7, sig, N)
x8 = np.random.multivariate_normal(u8, sig, N)

x = np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8])
y_true = np.atleast_2d(np.concatenate([np.zeros(4*N), np.ones(4*N)])).T

#Shuffle Data
"""For online training, the data is shuffled to reduce bias toward a single class during training"""
s = np.arange(x.shape[0])
np.random.shuffle(s)

#Implement Neural Net
jon_net = NeuralNetwork()
layer1 = Layer(2, 6)
layer2 = Layer(6, 4)
layer3 = Layer(4, 1)
jon_net.add_layer(layer1)
jon_net.add_layer(layer2)
jon_net.add_layer(layer3)

#Train Neural Net
mse, y_guess = jon_net.train(x[s], y_true[s], 0.05, 0.88, 150000)
x_pred = jon_net.predict(x)


#Plot Training Data Predictions
colors = ['green', 'blue']
fig = plt.figure(figsize=(13,13))
plt.scatter(x[:,0],x[:,1], c=x_pred[:,0], cmap=matplotlib.colors.ListedColormap(colors), marker='.', linewidths=3)
plt.show()

#Plot Error Curve
fig2 = plt.figure(figsize=(13,13))
x_grid = np.linspace(0, len(mse), len(mse))
plt.plot(x_grid, mse)
plt.show()

fig3 = plt.figure(figsize=(13,13))
x_grid2 = np.linspace(0, len(y_guess), len(y_guess))
plt.plot(x_grid2, y_guess)
plt.show()

t = np.linspace(0, 4, 1000)
t_grid = []
for i in t:
    for j in t:
        t_grid.append(np.array(i,j))



"""
#4.2 Redo

#Set up Mean Vectors and Covariance Matrix
u1 = np.array([0,0])
u2 = np.array([1,1])
u3 = np.array([0,1])
u4 = np.array([1,0])
sig = np.matrix([[0.01, 0.0], [0.0, 0.01]])
N = 50
S = 15000 #Max Iterations

#Produce sample vectors and assign to classes
x1 = np.random.multivariate_normal(u1, sig, N)
x2 = np.random.multivariate_normal(u2, sig, N)
x3 = np.random.multivariate_normal(u3, sig, N)
x4 = np.random.multivariate_normal(u4, sig, N)

x = np.concatenate([x1, x2, x3, x4])
y_true = np.atleast_2d(np.concatenate([np.zeros(2*N), np.ones(2*N)])).T

#Shuffle
s = np.arange(x.shape[0])
np.random.shuffle(s)

#Implement Neural Net 4.2
jon_net = NeuralNetwork()
layer1 = Layer(2, 2)
layer2 = Layer(2, 1)
jon_net.add_layer(layer1)
jon_net.add_layer(layer2)

#Train Neural Net 4.2
mse, y_guess = jon_net.train(x[s], y_true[s], 0.055, 0, 6000)

#Plot Error Curve
fig2 = plt.figure(figsize=(13,13))
x_grid = np.linspace(0, len(mse), len(mse))
plt.plot(x_grid, mse)
plt.show()

fig3 = plt.figure(figsize=(13,13))
x_grid2 = np.linspace(0, len(y_guess), len(y_guess))
plt.plot(x_grid2, y_guess)
plt.show()

#Test Neural Net
x_raw = jon_net.feed_forward(x)
x_pred = jon_net.predict(x)
print("Accuracy: " + str(jon_net.accuracy(x_pred, y_true)))

colors = ['green', 'blue']
fig = plt.figure(figsize=(13,13))
plt.scatter(x[:,0],x[:,1], c=x_pred[:,0], cmap=matplotlib.colors.ListedColormap(colors), marker='x', linewidths=3)
plt.show()

"""