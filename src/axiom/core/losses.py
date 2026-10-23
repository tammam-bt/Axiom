import numpy as np

def _choose_loss(name):
        if name in ["bce", "binarycrossentropy", "binary_cross_entropy"]:
            return (BCE, BCE_derivative)
        elif name in ["logcosh", "log_cosh", "log_hyperbolic_cosine"]:
            return (LOG_COSH, LOG_COSH_derivative)
        elif name in ["cce", "categorical_crossentropy"]:
            return (CCE, CCE_derivative)
        else:
            # MSE by default
            return (MSE, MSE_derivative)
    
# --- Loss Functions ---

def BCE( y_true, y_predicted):
    y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
    return np.mean(-(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted)))

def BCE_derivative( y_true, y_predicted):
    y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
    batch_size = y_true.shape[0]
    return (1 / batch_size) * ((y_predicted - y_true) / (y_predicted * (1 - y_predicted)))

def MSE( y_true, y_predicted):
    return np.mean((y_predicted - y_true)**2)

def MSE_derivative( y_true, y_predicted):
    batch_size = y_true.shape[0]
    return (2 / batch_size) * (y_predicted - y_true) 

def LOG_COSH( y_true, y_predicted):
    return np.mean(np.log(np.cosh(y_predicted - y_true)))

def LOG_COSH_derivative( y_true, y_predicted):
    batch_size = y_true.shape[0]
    return (1 / batch_size) * (np.tanh(y_predicted - y_true))    

def CCE( y_true, y_predicted):
    y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
    return np.mean(-(y_true * np.log(y_predicted)))

def CCE_derivative( y_true, y_predicted):
    y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
    batch_size = y_true.shape[0]
    return -(y_true / y_predicted) / batch_size