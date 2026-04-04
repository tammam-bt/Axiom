import numpy as np
from .activations import Tanh,Sigmoid,Relu,LeakyRelu,Selu
from .layers import Dense
from ..core.losses import _choose_loss

class Sequential:
    """Manages the stack of neural network layers."""
    def __init__(self, layers=None):
        self.layers = list(layers) if layers else []
        self._smart_initializer()

    def add(self, layer):
        self.layers.append(layer)
        self._smart_initializer()

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data  

    def backward(self, output_gradient, learning_rate, simplified_math=False):
        # If simplified_math is True, the loss gradient is already fused with the 
        # final activation derivative (e.g., BCE + Sigmoid = A - Y), so we skip the last layer.
        layers_to_backprop = self.layers[:-1][::-1] if simplified_math else self.layers[::-1]
            
        for layer in layers_to_backprop:
            output_gradient = layer.backward(output_gradient, learning_rate)
            
        return output_gradient

    def _smart_initializer(self):
        """Automatically assigns the optimal weight initializer based on the subsequent activation layer."""
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            
            if isinstance(current_layer, Dense):
                if isinstance(next_layer, (Tanh, Sigmoid)):
                    current_layer._initialize_weights("xavier")
                elif isinstance(next_layer, (Relu, LeakyRelu)):
                    current_layer._initialize_weights("he")   
                elif isinstance(next_layer, Selu):
                    current_layer._initialize_weights("lecun")


class Model:
    """Wraps a Sequential architecture with training loops and loss calculations."""
    def __init__(self, sequential, loss="MSE"): 
        self.Sequential = sequential
        self.loss, self.loss_derivative = _choose_loss(loss)

    def predict(self, X):
        return self.Sequential.forward(X)

    def fit(self, X_train, Y_train, epochs, lr):
        print_interval = max(1, epochs // 100)
        
        for epoch in range(epochs):
            output = self.Sequential.forward(X_train)
            error = self.loss(Y_train, output)
            gradient, simplified_math = self._calculate_initial_gradient(Y_train, output)
            
            self.Sequential.backward(gradient, lr, simplified_math=simplified_math)
            
            if epoch % print_interval == 0:
                print(f"Epoch: {epoch} - Error: {error:.4f}")      
    
    def _calculate_initial_gradient(self, y_true, y_predicted):
        """Determines whether to use the fused 'shortcut' gradient or standard backprop."""
        if self._is_simplified_math_combination():
            # The fused mathematical shortcut: (A - Y) / batch_size
            return ((y_predicted - y_true) / y_true.shape[0], True)
        
        return (self.loss_derivative(y_true, y_predicted), False)
    
    def _is_simplified_math_combination(self):
        """Checks if the architecture qualifies for a fused backward pass calculation."""
        last_layer = self.Sequential.layers[-1]
        is_bce_sigmoid = self.loss.__name__ == "BCE" and isinstance(last_layer, Sigmoid)
        # To Add CCE + Softmax
        return is_bce_sigmoid
