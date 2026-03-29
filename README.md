<img src="https://cdn-icons-png.flaticon.com/512/7747/7747363.png" alt="Axiom Library Logo" height="150">

Axiom
=====

> **A foundational Machine Learning and Neural Computing library implemented from first principles.**

**Axiom** is a high-performance, minimal framework designed to bridge the gap between mathematical theory and software engineering. Built entirely on **NumPy**, it provides a transparent implementation of both gradient-based architectures (Neural Networks, Regressions) and logic-based structures (Decision Trees, Random Forests).

Designed for engineers and researchers who demand a "glass-box" view of algorithmic internals, Axiom eliminates the overhead of production heavyweights while maintaining the rigor of an industrial-grade stack.

***

🏗️ Core Architecture
---------------------

Axiom is organized into distinct modules, allowing for a hybrid approach to machine learning:

### 🧠 Neural Engine (`axiom.nn`)

*   **Sequential API:** Build deep architectures effortlessly using a stackable layer interface.
    
*   **Smart Initialization:** Automated selection of **He**, **Xavier**, or **LeCun** strategies based on subsequent activation functions.
    
*   **Optimization Suite:** Momentum-based SGD for accelerated convergence and L2 Regularization for robust generalization.
    
*   **Activations:** Comprehensive support for `ReLU`, `Leaky_ReLU`, `Sigmoid`, `Tanh`, and `SELU`.
    

### 🌲 Logic Suite (`axiom.trees`)

*   **Decision Trees:** Recursive splitting logic utilizing **Information Gain** and **Entropy/Gini Impurity**.
    
*   **Ensemble Methods:** Native support for **Random Forests** and **Boosting** strategies.
    
*   **Categorical Handling:** Efficient processing of discrete decision boundaries without gradient dependency.
    

### 📈 Linear Systems (`axiom.linear`)

*   **Closed-Form Solutions:** Linear Regression via the **Normal Equation** for direct mathematical optimization.
    
*   **Iterative Solvers:** Logistic Regression implemented with optimized Gradient Descent kernels.
    

***

⚡ Quick Start
-------------

### Installation

Ensure you have the core numerical dependency installed:

Bash

    pip install numpy 

### Running the XOR Benchmark

Verify the Neural Engine’s convergence on non-linear boundaries:

Bash

    git clone https://github.com/tammam-bt/Axiom.git
    cd Axiom
    python test_xor.py 

***

📉 Example Usage
----------------

### Building a Neural Network

Python

    import axiom as ax
    import numpy as np
    
    # Define architecture
    network = ax.nn.Sequential([
        ax.nn.Dense(2, 8, momentum_beta=0.9),
        ax.nn.ReLU(),
        ax.nn.Dense(8, 1),
        ax.nn.Sigmoid()
    ])
    
    model = ax.Model(network, loss="BCE")
    model.fit(x_train, y_train, epochs=2000, lr=0.1) 

### Deploying a Decision Tree

Python

    from axiom.trees import DecisionTreeClassifier
    
    clf = DecisionTreeClassifier(max_depth=5, criterion="entropy")
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test) 

***

🛠️ Technical Specifications
----------------------------

| Component | Supported Features | Optimization Strategy | Mathematical Basis |
| :--- | :--- | :--- | :--- |
| **Optimizers** | SGD, Momentum | Velocity tracking, Weight Decay (L2) | Gradient Descent |
| **Loss Functions** | BCE, MSE, CCE, Log-Cosh | Fused Gradients, Epsilon Clipping | Information Theory / Calculus |
| **Initializers** | He, Xavier, LeCun | Distribution-aware variance scaling | Statistical Initialization |
| **Tree Logic** | ID3 / C4.5 Optimized | Recursive partitioning | Shannon Entropy / Gini |

🚀 Roadmap & Future Milestones
------------------------------

*   \[ \] **Vectorized Mini-batching:** Transition from full-batch to stochastic mini-batch processing for large-scale data.
    
*   \[ \] **Adaptive Optimizers:** Implementation of **Adam** and **RMSProp** for automated learning rate scaling.
    
*   \[ \] **Convolutional Kernels:** Expanding the Neural Engine to support spatial feature extraction (CNNs).
    
*   \[ \] **Serialization:** Native `.npz` support for saving and deploying trained model weights.
    

***

📜 Licensing
------------

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

***
