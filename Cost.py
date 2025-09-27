"""This Module gives the classes
for different Cost functions and their core functionalities

implemented are: 
    - evaluation in class.cost(y_prediction, y_actual)
    - derivative in class.derivat(y_prediction, y_actual)


"""
import numpy as np


class QuadraticCost:
    @staticmethod
    def cost(y_pred, y) -> np.ndarray:
        return 1/len(y_pred) * np.sum((y_pred - y) ** 2)

    @staticmethod
    def derivat(y_pred, y) -> np.ndarray:
        return y_pred - y
    
class BinaryCrossEntropy:
    @staticmethod
    def cost(y_pred, y) -> np.ndarray:
        return -1/len(y_pred) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def derivat(y_pred, y) -> np.ndarray:
        return -(y/(y_pred + 10**-100) - (1 - y)/(1 - y_pred + 10**-100))/y.shape[0]
    
class CategoricalCrossEntropy:
    @staticmethod
    def cost(y_pred, y) -> np.ndarray:
        return -1/len(y_pred) * np.sum(np.sum(y * np.log(y_pred)))
        
    @staticmethod
    def derivat(y_pred, y) -> np.ndarray:
        raise NotImplementedError
    
class HingeLoss:
    @staticmethod
    def cost(y_pred, y) -> np.ndarray:
        return np.mean(np.maximum(0, 1 - y * y_pred))
    
    @staticmethod
    def derivat(y_pred, y) -> np.ndarray:
        raise NotImplementedError