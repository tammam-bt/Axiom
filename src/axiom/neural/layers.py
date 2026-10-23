import numpy as np

class Layer:
    """Base class for all neural network layers."""
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        raise NotImplementedError("Forward method must be implemented by subclass.")
    
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Backward method must be implemented by subclass.")


class Dense(Layer):
    """Fully connected layer with momentum and L2 regularization."""
    def __init__(self, input_size, output_size, initialization="xavier", momentum_beta=0.9, l2_lambda=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Hyperparameters
        self.momentum_beta = momentum_beta
        self.l2_lambda = l2_lambda
        
        # Trainable parameters
        self.weights = None 
        self.bias = np.zeros((1, output_size))
        self.velocity = np.zeros((input_size, output_size)) # For momentum
        
        self._initialize_weights(initialization)
    
    def _initialize_weights(self, init_type):
        """Internal method to handle weight initialization strategies."""
        init_type = init_type.lower().strip()
        
        if init_type == "lecun":
            limit = np.sqrt(3 / self.input_size)
        elif init_type in ["he", "kaiming"]:
            limit = np.sqrt(6 / self.input_size)
        else:
            # Xavier/Glorot by default
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            
        self.weights = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        
    def forward(self, input_data):
        self.input = input_data
        # X: (Batch, Input_Size) dot W: (Input_Size, Output_Size) -> (Batch, Output_Size)
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Calculate gradients
        weights_gradient = np.dot(self.input.T, output_gradient)
        
        # Gradient to pass down to the previous layer
        back_prop = np.dot(output_gradient, self.weights.T)
        
        # Update velocity (Momentum)
        self.velocity = (self.momentum_beta * self.velocity) + ((1 - self.momentum_beta) * weights_gradient)
        
        # Update weights (with L2 decay) and biases
        self.weights -= learning_rate * (self.velocity + self.l2_lambda * self.weights)
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        
        return back_prop
    
class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self, activation_fn, derivative_fn):
        super().__init__()
        self.activation = activation_fn
        self.activation_derivative = derivative_fn

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Element-wise multiplication (Hadamard product)
        return output_gradient * self.activation_derivative(self.output)
    