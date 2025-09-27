"""This Module gives the class
for different Activation functions and their 
respective derivative in activation_prime()

example:
Activation.sigmoid(numpy.ndarray) -> numpy.ndarray
    returns the vector in which each element is 
    the projection of the sigmoid function to that element

Activation.sigmoid_prime(numpy.ndarray) -> numpy.ndarray
    returns the vector for the first derivative 
    of the sigmoid function 


"""
import numpy as np


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        return Activation.sigmoid(x)*(1.0-Activation.sigmoid(x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_prime(x):
        return 1 - Activation.tanh(x) ** 2

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_prime(x):
        return np.greater(x, 0.).astype(np.float64)
    
    @staticmethod
    def PReLU(m, x):
        return np.maximum(m*x, x)
    
    @staticmethod
    def PReLU_prime(m, x):
        raise NotImplementedError
    
    @staticmethod
    def ELU(alpha, x):
        raise NotImplementedError
    
    @staticmethod
    def ELU_prime(alpha, x):
        raise NotImplementedError
    
    @staticmethod
    def softmax(x, shift=True):
        if shift:
            norm = -np.max(x)
            return np.exp(x + norm) / np.sum(np.exp(x + norm))
        else: return np.exp(x) / np.sum(np.exp(x))
    
    @staticmethod
    def swish(x):
        return x * Activation.sigmoid(x)