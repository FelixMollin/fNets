"""This Module gives the class 
for a standard feed-forward Neural Network


"""
import numpy as np
import random
from .Cost import *
from .Activation import *


class FFNet:

    def __init__(self, size=None, types=None, weights=None, bias=None, cost=None):
        if weights: assert bias
        if bias: assert weights

        if size:
            self.size = size
            self.layercount = len(size)
            
            self.weights = [np.random.randn(x, y) 
                            for y, x in zip(size[:-1], size[1:])]
            self.biases = [np.zeros((y, 1)) 
                        for y in size[1:]]
            if types: self.types = types
            else: self.types = [Activation.sigmoid for _ in range(len(size))]
            if cost: self.cost = cost
            else: self.cost = QuadraticCost
        else: 
            assert types==None 
            self.size = []
            self.layercount = 0
            self.weights = []
            self.biases = []
            self.types = []
            if cost: self.cost = cost
            else: self.cost = QuadraticCost

    #adds to the end layer
    def add(self, size, type=None):
        if self.layercount == 0:
            self.size.append(size)
            self.layercount += 1
        else: 
            if type: self.types.append(type)
            else: self.types.append(Activation.sigmoid)
            self.weights.append(np.random.randn(size, self.size[-1]))
            self.biases.append(np.zeros((size, 1)))            
            self.size.append(size)
            self.layercount += 1

    def feedforward(self, X):
        for bias, weight, i in zip(self.biases, self.weights, range(self.layercount)):
            X = self.types[i](np.dot(weight, X) + bias)
        return X

    def train(self, X, y, batch_size, epochs, learning_rate):
        n = len(X)
        seed = random.randint(0, 100_000)

        x_batches = np.array([X[k:k+batch_size]
                        for k in range(0, n, batch_size)])
        y_batches = np.array([y[k:k+batch_size]
                        for k in range(0, n, batch_size)])
                
        for epoch in range(epochs):
            for x, y in zip(x_batches, y_batches):
                self.__backprop(x, y, len(x), learning_rate)

            print(f"Epoch {epoch} completed")

            # np.random.seed(seed)
            # np.random.shuffle(x_batches)
            # np.random.seed(seed)
            # np.random.shuffle(y_batches)

    def __backprop(self, X, y, len_X, learning_rate):
        learning_rate = learning_rate / len_X

        d_biases = [np.zeros(b.shape) for b in self.biases]
        d_weights = [np.zeros(w.shape) for w in self.weights]

        for X, y in zip(X, y):
            # feedforward 

            activation = X
            activations = [X]
            zs = []

            for bias, weight in zip(self.biases, self.weights):
                z = np.dot(weight, activation) + bias
                activation = Activation.sigmoid(z)
                zs.append(z)
                activations.append(activation)

            # Delta Outputlayer
            dL = np.multiply(self.cost.derivat(activations[-1], y), Activation.sigmoid_prime(zs[-1]))

            for l in range(2, self.layercount):
                # Delta in Layer l and saving previous delta to change self.biases and self.weights inplace
                prev_dL = dL
                dL = np.multiply(Activation.sigmoid_prime(zs[-l]), np.dot(self.weights[-l+1].T, dL))

                d_biases[-l+1] += prev_dL
                d_weights[-l+1] += np.dot(prev_dL, zs[-l].T)

                # self.biases[-l+1] -= prev_dL * learning_rate
                # self.weights[-l+1] -= np.dot(prev_dL, zs[-l].T) * learning_rate

            # self.biases[0] -= dL * learning_rate
            # self.weights[0] -= np.dot(zs[0].T, dL) * learning_rate

            d_biases[0] += dL
            d_weights[0] += np.dot(zs[0].T, dL)

            self.biases = [b - db * learning_rate
                           for b, db in zip(self.biases, d_biases)]
            self.weights = [w - dw * learning_rate
                            for w, dw in zip(self.weights, d_weights)]
