import numpy as np
class Layer:
    def __init__(self):
        self.input, self.output = None,None
    
    def forward(self, input):
        pass
    
    def backward(self, output_gradient, learning_rate):
        pass
class Dense(Layer):
    def __init__(self, input_size, output_size, initialization = "xavier", beta = 0.9):
        self.input_size = input_size
        self.output_size = output_size
        self.velocity = np.zeros((input_size,output_size))
        self.beta = beta
        self.weights_initializer(initialization)
        self.bias = np.zeros((1, output_size))
    
    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    # def backward(self, output_gradient, learning_rate):
        
    #     weights_gradient = np.dot(self.input.T, output_gradient)
    #     back_prop = np.dot(output_gradient, self.weights.T)
    #     self.weights -= learning_rate * weights_gradient
    #     self.bias -= learning_rate * np.sum(output_gradient, axis = 0, keepdims=True)
    #     return back_prop
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        self.velocity = self.beta * self.velocity + (1 - self.beta) * weights_gradient
        back_prop = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * self.velocity
        self.bias -= learning_rate * np.sum(output_gradient, axis = 0, keepdims=True)
        return back_prop
    
    def weights_initializer(self,initialization):
        if initialization.lower().strip() in ["lecun"]:
            limit = np.sqrt(3 / self.input_size)
            self.weights = np.random.uniform(-limit,limit,(self.input_size,self.output_size))
            
        elif initialization.lower().strip() in ["he","kaiming"]:
            limit = np.sqrt(6 / self.input_size)
            self.weights = np.random.uniform(-limit,limit,(self.input_size,self.output_size))
        else:
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = np.random.uniform(-limit,limit,(self.input_size,self.output_size))    

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
    def __init__(self, layers = None):
        self.layers = layers
        self.smart_initializer()
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
        self.smart_initializer()
    def smart_initializer(self):
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            if isinstance(current_layer, Dense):
                if isinstance(next_layer, (Tanh, Sigmoid)):
                    current_layer.weights_initializer("xavier")
                elif isinstance(next_layer, (ReLU, Leaky_ReLU)):
                    current_layer.weights_initializer("he")   
                elif isinstance(next_layer, SELU):
                    current_layer.weights_initializer("lecun")
                else:
                    pass    
            


class Model:
    def __init__(self,sequential, loss = "MSE"): 
        self.Sequential = sequential
        self.loss, self.loss_derivative = self.choose_loss(loss)
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
    
    def choose_loss(self,name):
        if name.lower().strip() in ["bce", "binarycrossentropy", "binary_cross_entropy"]:
            return (self.BCE, self.BCE_derivative)
        else:
            #By default MSE
            return (self.MSE, self.MSE_derivative)
    
    def BCE(self,y_true, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        return np.mean(y_true * np.log(y_predicted) + (1-y_true) * np.log(1 - y_predicted))

    def BCE_derivative(self,y_true, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        n = y_true.size
        return (1 / n) * ((y_predicted - y_true) / (y_predicted * (1 - y_predicted)))

    def MSE(self,y_true, y_predicted):
        return np.mean(((y_predicted-y_true)**2))

    def MSE_derivative(self,y_true, y_predicted):
        n = y_true.size
        return (2/n) * (y_predicted - y_true)        
                
                       
        

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

class ReLU(Activation):
    def __init__(self):
        relu = lambda x : np.maximum(0,x)
        relu_prime = lambda x : (x > 0).astype(float)
        super().__init__(relu, relu_prime)       

class Leaky_ReLU(Activation):
    def __init__(self, alpha = 0.01):
        leaky_relu = lambda x : np.maximum(alpha * x,x)
        leaky_relu_prime = lambda x : np.where(x > 0, 1, alpha)
        super().__init__(leaky_relu, leaky_relu_prime)         
        
class SELU(Activation):
    def __init__(self, _lambda = 1.0507, _alpha = 1.67326):
        selu = lambda x : np.where(x > 0, _alpha * _lambda * (np.exp(x) - 1), _lambda * x)
        selu_prime = lambda x : np.where(x > 0, _alpha * _lambda * np.exp(x), _lambda)
        super().__init__(selu, selu_prime)                 
          