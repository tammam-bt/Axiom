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

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)       

class Leaky_ReLU(Activation):
    def __init__(self, alpha=0.01):
        leaky_relu = lambda x: np.where(x > 0, x, alpha * x)
        leaky_relu_prime = lambda x: np.where(x > 0, 1.0, alpha)
        super().__init__(leaky_relu, leaky_relu_prime)        
        
class SELU(Activation):
    def __init__(self, _lambda=1.0507, _alpha=1.67326):
        selu = lambda x: np.where(x > 0, _lambda * x, _lambda * _alpha * (np.exp(x) - 1))
        selu_prime = lambda x: np.where(x > 0, _lambda, _lambda * _alpha * np.exp(x))
        super().__init__(selu, selu_prime)    


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
        
        if not simplified_math:
            output_gradient = self.layers[-1].backward(output_gradient, learning_rate) 
            
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
                elif isinstance(next_layer, (ReLU, Leaky_ReLU)):
                    current_layer._initialize_weights("he")   
                elif isinstance(next_layer, SELU):
                    current_layer._initialize_weights("lecun")


class Model:
    """Wraps a Sequential architecture with training loops and loss calculations."""
    def __init__(self, sequential, loss="MSE"): 
        self.Sequential = sequential
        self.loss_function_name = loss.lower().strip()
        self.loss, self.loss_derivative = self._choose_loss(self.loss_function_name)

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
        is_bce_sigmoid = self.loss_function_name in ["bce", "binarycrossentropy"] and isinstance(last_layer, Sigmoid)
        is_mse_linear = self.loss_function_name == "mse" and isinstance(last_layer, Dense)
        # Add CCE + Softmax check here later if needed
        return is_bce_sigmoid or is_mse_linear

    def _choose_loss(self, name):
        if name in ["bce", "binarycrossentropy", "binary_cross_entropy"]:
            return (self.BCE, self.BCE_derivative)
        elif name in ["logcosh", "log_cosh", "log_hyperbolic_cosine"]:
            return (self.LOG_COSH, self.LOG_COSH_derivative)
        elif name in ["cce", "categorical_crossentropy"]:
            return (self.CCE, self.CCE_derivative)
        else:
            # MSE by default
            return (self.MSE, self.MSE_derivative)
    
    # --- Loss Functions ---

    def BCE(self, y_true, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        return np.mean(-(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted)))

    def BCE_derivative(self, y_true, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        batch_size = y_true.shape[0]
        return (1 / batch_size) * ((y_predicted - y_true) / (y_predicted * (1 - y_predicted)))

    def MSE(self, y_true, y_predicted):
        return np.mean((y_predicted - y_true)**2)

    def MSE_derivative(self, y_true, y_predicted):
        batch_size = y_true.shape[0]
        return (2 / batch_size) * (y_predicted - y_true) 
    
    def LOG_COSH(self, y_true, y_predicted):
        return np.mean(np.log(np.cosh(y_predicted - y_true)))
    
    def LOG_COSH_derivative(self, y_true, y_predicted):
        batch_size = y_true.shape[0]
        return (1 / batch_size) * (np.tanh(y_predicted - y_true))    

    def CCE(self, y_true, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        return np.mean(-(y_true * np.log(y_predicted)))

    def CCE_derivative(self, y_true, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
        batch_size = y_true.shape[0]
        return -(y_true / y_predicted) / batch_size