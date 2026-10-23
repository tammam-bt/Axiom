import numpy as np
from ..core.base import BaseEstimator
from ..core.losses import BCE

class LogisticRegression(BaseEstimator):
    def __init__(self, b = 0.0, alpha = 0.01, verbose = False):
        self.w = None
        self.b = b
        self.alpha = alpha
        self.verbose = verbose
    
    def fit(self, X, y, epochs = 1000, with_logging = False):
        self.w = np.ones(X.shape[1])
        if with_logging:
            log = {"Weights": [[0 for i in range(X.shape[1])] for j in range(epochs)], "Bias": [0 for i in range(epochs)], "Cost": [0 for i in range(epochs)]}
        for i in range(epochs):
            if with_logging:
                log["Weights"][i] = self.w.copy()  # self.w is updated in place below
                log["Bias"][i] = self.b
                log["Cost"][i] = self.cost_function(X,y)
            if self.verbose:
                print(f"Iteration {i}: cost = {self.cost_function(X,y)}, bias = {self.b}")
            dc_dw,dc_db = self.calculate_gradient(X,y)
            self.w -= self.alpha * dc_dw
            self.b -= self.alpha * dc_db
        if with_logging:
            return log
        return None 
        
    
    def predict_proba(self, X):
        """Probability of class 1 for each row."""
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold = 0.5):
        """Class labels (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
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
        # Gradient of the class-weighted BCE. Each class gets the weight
        # m / (2 * its count), so both classes count equally whatever their sizes;
        # the weight multiplies the whole error p - y.
        m = X.shape[0]
        wp, wn = m / (2 * max(np.sum(y == 1), 1)), m / (2 * max(np.sum(y == 0), 1))
        error = np.where(y == 1, wp, wn) * (self.sigmoid(np.dot(X, self.w) + self.b) - y)
        dc_dw = X.T @ error / m
        dc_db = np.sum(error) / m
        return dc_dw, dc_db
       
    
    