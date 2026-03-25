KarthagoNet API Reference
This manual provides a technical overview of the classes and methods available in the KarthagoNet library.

Core Workflow
Define Layers: Create a list of Dense and Activation layers.

Initialize Sequential: Pass the layers into a Sequential container.

Build Model: Wrap the sequential object in a Model with a specified loss function.

Train: Use the fit method with your training data.

1. Sequential Layer Management
   The Sequential class handles the flow of data through the network and manages backpropagation.

Sequential(layers=None)
layers (list, optional): A list of layer objects.

add(layer): Appends a new layer to the stack.

Note: KarthagoNet features Smart Initialization. When you add an activation layer after a Dense layer, the weights of that Dense layer are automatically re-initialized using the optimal strategy (He for ReLU, Xavier for Sigmoid/Tanh, LeCun for SELU).

2. Layers
   Dense(input_size, output_size, initialization="xavier", momentum_beta=0.9, l2_lambda=0.0)
   A fully connected linear layer.

input_size: Number of incoming features.

output_size: Number of neurons in the layer.

initialization: "xavier", "he", or "lecun".

momentum_beta: Coefficient for momentum-based updates (default 0.9). Set to 0 for standard SGD.

l2_lambda: L2 regularization penalty (Weight Decay).

Activations
All activation layers inherit from the base Activation class and require no arguments unless specified.

Tanh()

Sigmoid()

ReLU()

Leaky_ReLU(alpha=0.01)

SELU(\_lambda=1.0507, \_alpha=1.67326)

3. The Model Class
   The Model class is the primary interface for training and inference.

Model(sequential, loss="MSE")
sequential: An instance of the Sequential class.

loss: String identifier for the loss function.

Supported: "MSE", "BCE", "CCE", "LOG_COSH".

Methods
predict(X)
Returns the network's output for a given input matrix.

X: NumPy array of shape (Batch_Size, Input_Features).

fit(X_train, Y_train, epochs, lr)
Trains the model using the provided data.

X_train: Training features.

Y_train: Training labels/targets.

epochs: Number of full passes over the dataset.

lr: Learning rate.

4. Quick Start Example (XOR Problem)
   Python
   import numpy as np
   import KarthagoNet as NN

# 1. Prepare Data

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

# 2. Define Architecture

network = NN.Sequential([
NN.Dense(2, 3), # Input: 2, Hidden: 3
NN.Sigmoid(),
NN.Dense(3, 1), # Hidden: 3, Output: 1
NN.Sigmoid()
])

# 3. Compile Model

model = NN.Model(network, loss="BCE")

# 4. Train

model.fit(X, Y, epochs=1000, lr=0.1)

# 5. Inference

predictions = model.predict(X)
print(predictions)
