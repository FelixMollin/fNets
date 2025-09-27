"""
This Module gives the class 
for a standard feed-forward Neural Network


"""
import numpy as np
import random
import threading

import json
import sys

from .Cost import *
from .Activation import *


class FFNet:

    def __init__(self, size=None, types=None, weights=None, biases=None, cost=None):
        if weights: assert biases
        if biases: assert weights

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

        self.data = []

    def add(self, size:int, type=Activation.sigmoid) -> None:
        """
        Appends a Layer of specified type Neurons to the end of the FFNet

        Parameters:
        ---
        size : int 
            - derermins the size of the to be appending layer

        type : Activation
            - if type is not specified sigmoid Activation will be chosen
        """
        if self.layercount == 0:
            self.size.append(size)
            self.layercount += 1
        else: 
            self.types.append(type)
            self.weights.append(np.random.randn(size, self.size[-1]))
            self.biases.append(np.zeros((size, 1)))            
            self.size.append(size)
            self.layercount += 1

    def feedforward(self, X:np.ndarray) -> np.ndarray:
        """
        Evaluates FFNet on the given Data 

        Parameters:
        ---
        X : numpy.ndarray
            - ndarray of dim <= 2 and shape needs to match inputlayer shape

        Notes:
        feedforward does not check for if X.shape is the same as 
        the desired shape for the inputlayer, thus shape (x,) or (1,x)
        may need to be transposed with X.T or X.transpose() before passing 
        as a Parameter
        
        """
        for bias, weight, i in zip(self.biases, self.weights, range(self.layercount)):
            X = self.types[i](np.dot(weight, X) + bias)
        return X

    def train(self, X:np.ndarray, y:np.ndarray, batch_size:int, epochs:int, learning_rate:float, train_all=False) -> None:
        """
        Trains the Network on the provided Data 
        
        Parameters:
        ---
        X : numpy.ndarray
            - ndarray of dim <= 3 and shape of individual Data 
              needs to match inputlayer shape

        y : numpy.ndarray
            - ndarray of dim <= 2 with each index representing
              the y Value of the coresponding X index

        batch_size : int 
            - determines the size of the Batches in which the 
              data will be devided, with batch_size=1 online 
              learning is achieved
            
        learning_rate : float
            - determines learning rate for gradient descent 

        train_all : boolean 
            - to assess wether to train on the entire data the
              model has received or just the given data, 
              default is False to shorten computation

        Notes:
        - batch_size needs to be chosen with care as it needs to devide 
          the input data with no remainder
        - train_all has no affect at the moment, implementation will follow
        - the shuffeling of the training batches is not implemented at the moment

        
        """
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

    def integrate(self, X:np.ndarray, y:np.ndarray, batch_size:int, epochs:int, learning_rate:float, thread=True) -> tuple:
        """
        Integrates new Data into an existing Network, without stoping it

        Parameters:
        ---
        X : numpy.ndarray
            - ndarray of dim <= 3 and shape of individual Data 
              needs to match inputlayer shape

        y : numpy.ndarray
            - ndarray of dim <= 2 with each index representing
              the y Value of the coresponding X index

        batch_size : int 
            - determines the size of the Batches in which the 
              data will be devided, with batch_size=1 online 
              learning is achieved
            
        learning_rate : float
            - determines learning rate for gradient descent 

        thread : boolean
            - default True to start thread/learning before returning 

            

        Return : tuple -> (old_Network, train_thread)
            - old_Network will be a new instance of FFNet so that the old Network can 
              still be used while the new one trains the Network inplace

        Notes:
        - /
    
        """
        train_thread = threading.Thread(target=self.train, args=(X, y, batch_size, epochs, learning_rate))
        old_Network = FFNet(self.size, types=self.types, weights=self.weights, biases=self.biases, cost=self.cost)
        if thread: train_thread.start() 
        return (old_Network, train_thread)

    def __backprop(self, X:np.ndarray, y:np.ndarray, len_X:int, learning_rate:float) -> None:
        """
        implementation of gradient descent 

        Notes:
            - weights/biases will be changed inplace 
        
        """
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

    def save(self, file:str):
        """
        save FFNet fields to specidied .json

        Parameters:
            - file gives the file name 

        Notes:
            - types are not saved at the moment
        """
        data = {"size": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(file, "w")
        json.dump(data, f)
        f.close()

    @staticmethod
    def open(file:str):
        """
        load FFNet and return it 

        Parameters:
            - file gives the file name 

        Notes:
            - types are not saved at the moment
        """
        f = open(file, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        # types = getattr(sys.modules[__name__], data["types"])
        return FFNet(data["size"], weights=data["weights"], biases=data["biases"], cost=cost)