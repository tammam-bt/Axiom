import numpy as np
import matplotlib.pyplot as plt



def MSE(y_true, y_predicted):
    return np.mean(((y_predicted-y_true)**2))

def MSE_derivative(y_true, y_predicted):
    n = y_true.size
    return (2/n) * (y_predicted - y_true)

class Layer:
    def __init__(self):
        self.input, self.output = None,None
    
    def forward(self, input):
        pass
    
    def backward(self, output_gradient, learning_rate):
        pass
class Dense(Layer):
    def __init__(self, input_size, output_size ):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit,limit,(input_size,output_size))
        self.bias = np.zeros((1, output_size))
    
    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        back_prop = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis = 0, keepdims=True)
        return back_prop

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_derivative(self.output)

class Sequential:
    def __init__(self, *args):
        self.layers = list(args)
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input  
    def backward(self, output_gradient, learning_rate):
        for layer in self.layers[::-1]:
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient
    def add(self, layer):
        self.layers += layer


class Model:
    def __init__(self,sequential, loss = "MSE"): 
        self.Sequential = sequential
        self.loss = MSE
        self.loss_derivative = MSE_derivative
    def predict(self, X):
        return self.Sequential.forward(X)
    def fit(self, X_train, Y_train, epochs, lr):
        for epoch in range(epochs):
            output = self.Sequential.forward(X_train)
            error = self.loss(Y_train, output)
            gradient =  self.loss_derivative(Y_train, output) 
            self.Sequential.backward(gradient,lr)
            if(epoch % 10 == 0):
                print(f"Epoch : {epoch} - Error : {error}")      
            
                
                       
        

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x : np.tanh(x)
        tanh_prime = lambda x : 1 - x**2
        super().__init__(tanh,tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x : np.clip(1 / (1 + np.exp(-x)), 1e-15, 1 - 1e-15)
        sigmoid_prime = lambda x: x * (1 - x)
        super().__init__(sigmoid,sigmoid_prime)        
          