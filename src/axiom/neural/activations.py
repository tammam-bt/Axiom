from .layers import Activation
import numpy as np
# --- Specific Activation Functions ---

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - x**2 # Assumes x is the activated output
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: np.clip(1 / (1 + np.exp(-x)), 1e-15, 1 - 1e-15)
        sigmoid_prime = lambda x: x * (1 - x) # Assumes x is the activated output
        super().__init__(sigmoid, sigmoid_prime)  

class Relu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)       

class LeakyRelu(Activation):
    def __init__(self, alpha=0.01):
        leaky_relu = lambda x: np.where(x > 0, x, alpha * x)
        leaky_relu_prime = lambda x: np.where(x > 0, 1.0, alpha)
        super().__init__(leaky_relu, leaky_relu_prime)        
        
class Selu(Activation):
    def __init__(self, _lambda=1.0507, _alpha=1.67326):
        selu = lambda x: np.where(x > 0, _lambda * x, _lambda * _alpha * (np.exp(x) - 1))
        selu_prime = lambda x: np.where(x > 0, _lambda, _lambda * _alpha * np.exp(x))
        super().__init__(selu, selu_prime)  