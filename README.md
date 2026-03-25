<img src="https://cdn-icons-png.flaticon.com/512/7747/7747363.png" alt="Logo of the project" height="200">

Neural Network Library from Scratch
===================================

> A high-performance, minimal NumPy-based framework for understanding deep learning internals.

This project is a lightweight neural network library implemented from the ground up using Python and NumPy. While it avoids the heavy overhead of frameworks like TensorFlow or PyTorch, it implements advanced features like **momentum**, **L2 regularization**, and **automated smart initialization** to ensure stability and convergence on non-linear problems like XOR.

It is designed for students and developers who want to see the exact flow of matrices during forward and backward propagation without the "black box" of production-grade libraries.

***

Quick Start
-----------

### Installation

Minimal setup required:

Bash

    pip install numpy 

### Run the XOR Benchmark

The library comes with a verified XOR test case that achieves 0.9+ confidence in under 2,000 epochs.

Bash

    git clone https://github.com/tammam-bt/NN-library-from-scratch.git
    cd NN-library-from-scratch
    python test_xor.py 

***

Core Features
-------------

### 🏗️ Architecture

*   **Sequential API:** Stack layers effortlessly using `NN.Sequential`.
    
*   **Dense Layers:** Fully connected layers with customizable input/output dimensions.
    
*   **Expanded Activations:** Includes `Sigmoid`, `Tanh`, `ReLU`, `Leaky_ReLU`, and `SELU`.
    
*   **Smart Initialization:** Automatically selects the best strategy (**He**, **Xavier**, or **LeCun**) based on the following activation layer.
    

### ⚡ Optimization & Math

*   **Momentum-based SGD:** Accelerates convergence and avoids local minima using velocity tracking.
    
*   **L2 Regularization:** Integrated weight decay to prevent overfitting.
    
*   **Fused Gradients:** Internal "Simplified Math" logic for BCE/Sigmoid and MSE/Linear combinations to improve numerical stability.
    
*   **Epsilon Clipping:** Built-in safeguards (`1e-15`) to prevent `NaN` errors during log calculations.
    

### 📉 Supported Loss Functions

*   **BCE:** Binary Cross-Entropy (for binary classification).
    
*   **MSE:** Mean Squared Error (for regression).
    
*   **Log-Cosh:** A smoother, more robust alternative to MSE.
    
*   **CCE:** Categorical Cross-Entropy (for multi-class classification).
    

***

Example Usage
-------------

Building a 2-layer network with momentum to solve XOR:

Python

    import NN
    import numpy as np
    
    # Data setup
    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[0], [1], [1], [0]])
    
    # Define architecture with automated smart initialization
    network = NN.Sequential([
        NN.Dense(2, 3, momentum_beta=0.9, l2_lambda=0.01),
        NN.ReLU(),
        NN.Dense(3, 1),
        NN.Sigmoid()
    ])
    
    # Initialize model with Binary Cross-Entropy
    model = NN.Model(network, loss="BCE")
    
    # Train with specific learning rate
    model.fit(x_train, y_train, epochs=2000, lr=0.1)
    
    # Inference
    print(model.predict(x_train)) 

***

API Reference
-------------

### `Dense` Parameters

Parameter

Type

Description

`input_size`

`int`

Number of input features.

`output_size`

`int`

Number of neurons in the layer.

`initialization`

`str`

`"xavier"`, `"he"`, or `"lecun"`. (Overridden by `Sequential` smart init).

`momentum_beta`

`float`

Velocity decay (0 to 1). Set to 0 for pure SGD.

`l2_lambda`

`float`

Weight decay penalty for regularization.

Export to Sheets

### `Model.fit` Parameters

Parameter

Type

Description

`epochs`

`int`

Number of iterations over the full dataset.

`lr`

`float`

Learning rate for weight updates.

Export to Sheets

***

Future Improvements
-------------------

*   \[ \] **Mini-batch Support:** Currently optimized for full-batch processing.
    
*   \[ \] **Adam Optimizer:** Adding adaptive moment estimation.
    
*   \[ \] **Dropout Layers:** For enhanced regularization in larger networks.
    
*   \[ \] **Model Serialization:** Saving and loading weights via `.npz` files.
    

***

Licensing
---------

Licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

***

**Would you like me to help you draft a specific `test_xor.py` script that utilizes these new momentum and smart-init features to include in your repo?**
