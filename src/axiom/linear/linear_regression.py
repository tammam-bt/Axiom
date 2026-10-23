import numpy as np
from ..core.base import BaseEstimator
from ..core.losses import MSE

class LinearRegression(BaseEstimator):
    def __init__(self, alpha = 0.1, verbose = False):
        self.alpha = alpha
        self.verbose = verbose
        self.weights = None
        self.bias = 0
    
    def fit(self, x, y, epochs = 1000, with_logging =False):
        if with_logging:
            cost_iterations = []
        n_features = x.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for i in range(epochs):
            dc_dw,dc_db = self.compute_gradient(x,y)
            self.weights -= self.alpha * dc_dw
            self.bias -= self.alpha * dc_db
            if self.verbose or with_logging:
                cost = self.cost_function(x,y)
            if self.verbose:
                print(f"Iteration {i}: cost = {cost}")
            if with_logging:
                cost_iterations.append(cost)
        if with_logging:
            return cost_iterations    
        return None
        
    def cost_function(self, x , y):
        return MSE(np.dot(x, self.weights) + self.bias, y)
    
    def compute_gradient(self, x, y):
        # Gradient of half the MSE, J = (1/2m) * sum((Xw + b - y)^2):
        #   dJ/dw = X^T (Xw + b - y) / m,   dJ/db = sum(Xw + b - y) / m
        # (the gradient of the MSE itself is twice this; the factor 2 is absorbed by alpha)
        m = x.shape[0]
        error = np.dot(x,self.weights) + self.bias - y
        dc_dw = x.T @ error / m
        dc_db = np.sum(error) / m
        return dc_dw,dc_db
    
    def predict(self, x):
        return np.dot(x, self.weights) + self.bias 
        