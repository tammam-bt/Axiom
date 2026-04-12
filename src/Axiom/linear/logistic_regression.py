import numpy as np
from ..core.base import BaseEstimator
from ..core.losses import BCE

class LogisticRegression(BaseEstimator):
    def __init__(self, b = 0.0, alpha = 0.01):
        self.w = None
        self.b = b
        self.alpha = alpha
    
    def fit(self, X, y, epochs = 1000, with_logging = False):
        self.w = np.ones(X.shape[1])
        if with_logging:
            log = {"Weights": [[0 for i in range(X.shape[1])] for j in range(epochs)], "Bias": [0 for i in range(epochs)], "Cost": [0 for i in range(epochs)]}
        for i in range(epochs):
            if with_logging:
                log["Weights"][i] = self.w
                log["Bias"][i] = self.b
                log["Cost"][i] = self.cost_function(X,y)
            print(f"Iteration {i}: cost = {self.cost_function(X,y)}, bias = {self.b}")
            dc_dw,dc_db = self.calculate_gradient(X,y)
            self.w -= self.alpha * dc_dw
            self.b -= self.alpha * dc_db
        if with_logging:
            return log
        return None 
        
    
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)
    
    def sigmoid(self,z):
        return np.clip(1 / (1 + np.exp(-z)), 1e-15, 1 - 1e-15) 
    
    def cost_function(self, X , y):
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        p = self.sigmoid(z)
        logp = np.log(p)
        loginversep = (np.log(1 - p))
        total_cost = (-1/m) * np.sum((y * logp + (1-y) * loginversep))
        return total_cost
    def calculate_gradient(self, X, y):
        dc_dw, dc_db = np.zeros(X.shape[1]), 0
        m,n = X.shape
        wp,wn = m/(2*np.sum(y==1)),m / (2*np.sum(y==0))
        error = np.where(y == 1, wp * self.sigmoid(np.dot(X, self.w) + self.b) - y, wn * self.sigmoid(np.dot(X, self.w) + self.b) - y)
        dc_dw = np.array([np.dot(error ,X[:,j]) / X.shape[0] for j in range(X.shape[1])])
        dc_db = np.sum(error)
        return dc_dw, dc_db
       
    
    