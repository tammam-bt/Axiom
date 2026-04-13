# Axiom (`axiom-ml`) — Technical Documentation

> **Version:** 0.1.0 | **Author:** Tammam BenBettayeb | **Python:** 3.8+ | **License:** MIT | **Dependency:** NumPy only

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
   - [Forward Propagation](#21-forward-propagation)
   - [Backpropagation & The Chain Rule](#22-backpropagation--the-chain-rule)
   - [Weight Initialization](#23-weight-initialization)
   - [Optimization](#24-optimization)
   - [Regularization](#25-regularization)
3. [API Reference](#3-api-reference)
   - [Model & Sequential](#31-model--sequential)
   - [Layers](#32-layers)
   - [Activations](#33-activations)
   - [Loss Functions](#34-loss-functions)
   - [Optimizers](#35-optimizers)
   - [Classical Algorithms](#36-classical-algorithms)
   - [Tree Ensemble Methods](#37-tree-ensemble-methods)
   - [Preprocessing](#38-preprocessing)
4. [End-to-End Examples](#4-end-to-end-examples)
5. [Extending Axiom](#5-extending-axiom)

---

## 1. Architecture Overview

Axiom is organized into four subpackages, each responsible for a distinct area of machine learning. All public symbols are re-exported from the top-level `axiom` namespace via `__init__.py`, so you never need to import from deep module paths unless you want to.

```
axiom/
├── core/        → base classes, loss functions, optimizers
├── neural/      → engine (Sequential/Model), Dense layer, activations
├── linear/      → LinearRegression, LogisticRegression
├── trees/       → DecisionTreeClassifier + ensemble methods
└── preprocessing/ → scalers, encoders
```

### Component Hierarchy During a Training Step

```
Sequential / Model  (training engine — neural/engine.py)
│
├── Dense      →  Z = XW + b         (neural/layers.py)
├── Relu       →  A = max(0, Z)       (neural/activations.py)
├── Dense      →  Z = AW + b
└── [output activation]
     │
     └── Loss Function  →  scalar J   (core/losses.py)
          │
          └── Optimizer  →  W := W - α·∇W   (core/optimizers.py)
```

### The Training Loop

At the core of every training run is a two-phase cycle:

**Phase 1 — Forward Pass:** Data flows left-to-right through the layer stack. Each layer transforms its input and caches the intermediate values needed for the backward pass.

**Phase 2 — Backward Pass:** The error signal flows right-to-left. Each layer receives the gradient from the layer ahead of it, computes its own local gradient using the cached values, and passes a gradient back to the layer behind it.

This clean separation means each component only needs to know about its **immediate neighbours**, not the full graph — a design that makes Axiom easily extensible.

---

## 2. Mathematical Foundations

This section documents the calculus and linear algebra that every component in Axiom implements. All operations are vectorized across a batch of `m` samples.

### 2.1 Forward Propagation

For a network with `L` layers, the forward pass computes:

```
Z[l] = A[l-1] · W[l] + b[l]      # Linear transformation
A[l] = g[l](Z[l])                  # Activation function applied element-wise
```

Where:

- `A[0] = X` — the input matrix of shape `(m, n_features)`
- `W[l]` — weight matrix of shape `(n[l-1], n[l])`
- `b[l]` — bias vector of shape `(1, n[l])`, broadcast across the batch
- `g[l]` — the activation function for layer `l`
- `A[L]` — the final output (predictions), shape `(m, n_outputs)`

### 2.2 Backpropagation & The Chain Rule

Backpropagation is an efficient application of the **chain rule** of calculus. Given a scalar loss `J`, we want to compute `∂J/∂W[l]` and `∂J/∂b[l]` for every layer `l`.

**Step 1 — Output layer gradient (classification example with cross-entropy + softmax):**

```
dA[L] = A[L] - Y      # shape: (m, n_outputs)
```

This is the elegant result of differentiating the softmax + cross-entropy combination together, which cancels many terms.

**Step 2 — Propagate through each layer (backward, from L to 1):**

```
dZ[l] = dA[l] * g'[l](Z[l])       # Element-wise: activation derivative
dW[l] = (1/m) * A[l-1].T · dZ[l]  # Weight gradient, averaged over batch
db[l] = (1/m) * sum(dZ[l], axis=0) # Bias gradient, averaged over batch
dA[l-1] = dZ[l] · W[l].T           # Gradient to pass to the previous layer
```

Every layer in Axiom implements a `backward(dA)` method that carries out exactly this computation, using the `Z[l]` and `A[l-1]` values cached during the forward pass.

### 2.3 Weight Initialization

Initializing all weights to zero causes **symmetry breaking failure** — every neuron learns the same thing. Initializing with naively large values causes **exploding gradients**. Axiom implements two principled strategies:

**Xavier / Glorot Initialization** (recommended for Sigmoid / Tanh activations):

```
W ~ Uniform(-√(6 / (n_in + n_out)),  +√(6 / (n_in + n_out)))
```

This preserves variance of activations across layers, derived from the condition:
`Var(W) = 2 / (n_in + n_out)`

**He Initialization** (recommended for ReLU-family activations):

```
W ~ Normal(0, √(2 / n_in))
```

Because ReLU kills half of its inputs (the negative half), He initialization compensates by using a larger variance: `Var(W) = 2 / n_in`. This is also the correct choice for **Leaky ReLU** and **SELU**, which are ReLU variants.

### 2.4 Optimization

#### Vanilla SGD

The simplest update rule: move each parameter in the direction opposite to its gradient, scaled by the learning rate `α`:

```
W := W - α * ∂J/∂W
b := b - α * ∂J/∂b
```

The problem with vanilla SGD is that gradients computed on small batches are **noisy**. This causes erratic, oscillatory updates, especially in directions of high curvature.

#### Momentum SGD

Momentum smooths out the noisy gradient signal by maintaining an **exponentially weighted moving average** of past gradients:

```
v_W := β * v_W + (1 - β) * ∂J/∂W     # Velocity accumulation
W   := W - α * v_W                      # Parameter update
```

Where `β` (typically `0.9`) controls how much history to retain. Intuitively, the velocity `v_W` acts like a "ball rolling downhill" — it builds momentum in consistent gradient directions and dampens oscillations in inconsistent ones, leading to faster and more stable convergence.

### 2.5 Regularization

#### L2 Regularization (Weight Decay)

Without regularization, the model can fit training noise by growing its weights arbitrarily large. L2 regularization adds a penalty term to the loss:

```
J_regularized = J + (λ / 2m) * Σ ||W[l]||²_F
```

Where `||W[l]||²_F` is the Frobenius norm (sum of squared elements) of the weight matrix for layer `l`, and `λ` is the regularization strength hyperparameter.

The gradient update for the weights becomes:

```
∂J_reg/∂W[l] = ∂J/∂W[l] + (λ/m) * W[l]
```

This extra term `(λ/m) * W[l]` effectively **shrinks the weights at every step** (hence "weight decay"), making it harder for the network to memorize and overfit the training data.

---

## 3. API Reference

All classes below are accessible directly from the top-level `axiom` package. Deep imports from submodules also work and are shown alongside each class.

### 3.1 Model & Sequential

```python
from axiom import Sequential, Model
# or: from axiom.neural.engine import Sequential, Model
```

Axiom exposes two training engine interfaces. `Sequential` is the higher-level, Keras-style API with a `compile` step. `Model` provides lower-level control where the loss and optimizer are passed at construction time.

#### `Sequential`

```python
model = Sequential()
model.add(Dense(64, input_dim=784))
model.add(Relu())
model.add(Dense(10))

model.compile(loss=CCE(), optimizer=MomentumSGD(lr=0.01))
history = model.fit(X_train, y_train, epochs=100, batch_size=32)
predictions = model.predict(X_test)
```

#### `Model`

```python
model = Model(loss=MSE(), optimizer=SGD(lr=0.05))
model.add(Dense(32, input_dim=8))
model.add(Relu())
model.add(Dense(1))

model.fit(X_train, y_train, epochs=200)
```

#### Common Methods

| Method                                | Description                                     |
| ------------------------------------- | ----------------------------------------------- |
| `model.add(layer)`                    | Append a layer or activation to the stack       |
| `model.compile(loss, optimizer)`      | _(Sequential only)_ Set loss and optimizer      |
| `model.fit(X, y, epochs, batch_size)` | Train the model; returns loss history           |
| `model.predict(X)`                    | Forward pass; returns raw output of final layer |
| `model.evaluate(X, y)`                | Returns scalar loss on a dataset                |

> **Note:** `input_dim` only needs to be specified on the **first** `Dense` layer. All subsequent layers infer their input size automatically.

---

### 3.2 Layers

```python
from axiom import Dense
# or: from axiom.neural.layers import Dense
```

#### `Dense(units, input_dim, initializer)`

A fully connected layer. Computes `Z = XW + b`.

| Parameter     | Type  | Default | Description                                          |
| ------------- | ----- | ------- | ---------------------------------------------------- |
| `units`       | `int` | —       | Number of neurons (output dimension)                 |
| `input_dim`   | `int` | `None`  | Input dimension (only required for the first layer)  |
| `initializer` | `str` | `"he"`  | Weight init strategy: `"he"`, `"xavier"`, `"random"` |

**Attributes after build:**

- `layer.W` — weight matrix, shape `(input_dim, units)`
- `layer.b` — bias vector, shape `(1, units)`
- `layer.dW` — weight gradient (populated after `backward()`)
- `layer.db` — bias gradient (populated after `backward()`)

**Core methods:**

- `layer.forward(A_prev)` → `Z` — linear transform, caches `A_prev` and `Z`
- `layer.backward(dZ)` → `dA_prev` — computes `dW`, `db`, returns upstream gradient

---

### 3.3 Activations

```python
from axiom import Relu, LeakyRelu, Sigmoid, Tanh, Selu
# or: from axiom.neural.activations import Relu, LeakyRelu, Sigmoid, Tanh, Selu
```

Activation functions are **stateless layers** — they implement `forward()` and `backward()` but hold no trainable parameters.

#### `Relu`

```
forward:   A = max(0, Z)
backward:  dZ = dA * (Z > 0)      # 1 where Z > 0, else 0
```

The standard choice for hidden layers. Computationally cheap and avoids vanishing gradients in the positive region. Use **He initialization** with ReLU.

#### `LeakyRelu`

```
forward:   A = Z if Z > 0 else α·Z        (α typically 0.01)
backward:  dZ = dA * (1 if Z > 0 else α)
```

Fixes the "dying ReLU" problem by allowing a small, non-zero gradient for negative inputs. Neurons can no longer become permanently inactivated.

#### `Sigmoid`

```
forward:   A = 1 / (1 + exp(-Z))
backward:  dZ = dA * A * (1 - A)
```

Best for **binary output layers**. Can suffer from **vanishing gradients** in deep networks because the derivative `A·(1−A)` is at most `0.25`, causing gradients to shrink with depth.

#### `Tanh`

```
forward:   A = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))
backward:  dZ = dA * (1 - A²)
```

Zero-centered, which makes it preferable to sigmoid for hidden layers in shallow networks. Still prone to vanishing gradients for very deep architectures.

#### `Selu`

```
forward:   A = λ · (Z if Z > 0 else α·(exp(Z) - 1))
           λ ≈ 1.0507,  α ≈ 1.6733  (fixed constants)
backward:  dZ = dA * (λ if Z > 0 else λ·α·exp(Z))
```

A self-normalizing activation: under the right weight initialization, SELU drives activations toward zero mean and unit variance automatically, making batch normalization unnecessary. Use **LeCun Normal initialization** with SELU.

---

### 3.4 Loss Functions

```python
from axiom import MSE, BCE, CCE, LOG_COSH
# or: from axiom.core.losses import MSE, BCE, CCE, LOG_COSH
```

#### `MSE` — Mean Squared Error

```
J = (1/m) * Σ (ŷ - y)²
dA = (2/m) * (ŷ - y)
```

Use for **regression** tasks where the output is a continuous value.

#### `BCE` — Binary Cross-Entropy

```
J = -(1/m) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
dA = -(y/ŷ) + (1-y)/(1-ŷ)
```

Use for **binary classification** (single sigmoid output neuron).

#### `CCE` — Categorical Cross-Entropy

```
J = -(1/m) * Σ_i Σ_k y_ik · log(ŷ_ik)
dA[L] = ŷ - y      # Simplified result of softmax + CCE combined
```

Use for **multi-class classification**. The gradient shown assumes this loss is always paired with a softmax output layer, which yields the clean analytical form above.

#### `LOG_COSH` — Log-Cosh Loss

```
J = (1/m) * Σ log(cosh(ŷ - y))
dA = (1/m) * tanh(ŷ - y)
```

A smooth approximation to MAE. Behaves like MSE for small errors and like MAE for large ones, making it **robust to outliers** without the non-differentiability of MAE at zero.

---

### 3.5 Optimizers

```python
from axiom.core.optimizers import SGD, MomentumSGD
```

#### `SGD(lr)`

| Parameter | Type    | Default | Description   |
| --------- | ------- | ------- | ------------- |
| `lr`      | `float` | `0.01`  | Learning rate |

#### `MomentumSGD(lr, beta)`

| Parameter | Type    | Default | Description          |
| --------- | ------- | ------- | -------------------- |
| `lr`      | `float` | `0.01`  | Learning rate        |
| `beta`    | `float` | `0.9`   | Momentum coefficient |

---

### 3.6 Classical Algorithms

#### Linear Regression

```python
from axiom import LinearRegression
# or: from axiom.linear.linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = model.score(X_test, y_test)
```

#### Logistic Regression

```python
from axiom import LogisticRegression
# or: from axiom.linear.logistic_regression import LogisticRegression

model = LogisticRegression(learning_rate=0.1, epochs=500)
model.fit(X_train, y_train)
preds = model.predict(X_test)       # Returns class labels
probs = model.predict_proba(X_test) # Returns sigmoid probabilities
acc   = model.score(X_test, y_test)
```

#### Decision Tree Classifier

```python
from axiom import DecisionTreeClassifier
# or: from axiom.trees.decision_tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=10, criterion="gini")
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
```

`criterion` options: `"gini"` (Gini impurity) or `"entropy"` (information gain). The splitting logic and gain computations are isolated in `axiom/trees/gains.py`, making it straightforward to audit or extend.

---

### 3.7 Tree Ensemble Methods

All ensemble methods live in `axiom/trees/` and are built on top of the same decision tree primitives.

#### Gradient Boosting Machine (GBM)

```python
from axiom.trees.gbm import GBM

model = GBM(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

GBM fits an additive sequence of weak learners (shallow trees), where each new tree is trained on the **pseudo-residuals** (negative gradients of the loss) from the current ensemble.

#### XGBoost

```python
from axiom.trees.xgboost import XGBoost

model = XGBoost(n_estimators=100, learning_rate=0.1, max_depth=4, lambda_=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

Extends GBM by using **second-order Taylor expansions** of the loss for more accurate gradient steps, and adds L2 regularization (`lambda_`) directly on the leaf weights.

#### LightGBM

```python
from axiom.trees.lightgbm import LightGBM

model = LightGBM(n_estimators=100, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

Uses **histogram-based splitting** and **leaf-wise (best-first) tree growth** rather than depth-wise growth, which reduces memory and speeds up training on larger datasets.

#### Linear Tree

```python
from axiom.trees.linear_tree import LinearTree

model = LinearTree(max_depth=4)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

A hybrid model: the tree structure partitions the feature space, but each leaf node fits a **linear regression** rather than a constant. This captures both non-linear boundaries (via splits) and smooth trends within each region.

---

### 3.8 Preprocessing

```python
from axiom.preprocessing.scalers import MinMaxScaler, StandardScaler
from axiom.preprocessing.encoders import LabelEncoder, OneHotEncoder
```

#### Scalers

```python
scaler = StandardScaler()          # Zero mean, unit variance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # Uses stats from training set only

scaler = MinMaxScaler()            # Scales to [0, 1]
X_scaled = scaler.fit_transform(X)
```

> **Important:** Always call `fit_transform` on training data and `transform` (never `fit_transform`) on test/validation data. Fitting on test data leaks information and inflates performance metrics.

#### Encoders

```python
# Label encoding (ordinal integers)
enc = LabelEncoder()
y_encoded = enc.fit_transform(["cat", "dog", "cat", "bird"])
# → [1, 2, 1, 0]

# One-hot encoding
ohe = OneHotEncoder()
y_onehot = ohe.fit_transform(y_encoded)
# → [[0,1,0], [0,0,1], [0,1,0], [1,0,0]]
```

---

## 4. End-to-End Examples

### 4.1 Multi-Class Classification (Iris Dataset)

```python
import numpy as np
from sklearn.datasets import load_iris
from axiom import Sequential, Dense, Relu, CCE
from axiom.neural.activations import Sigmoid
from axiom.core.optimizers import MomentumSGD
from axiom.preprocessing.scalers import StandardScaler
from axiom.preprocessing.encoders import OneHotEncoder

# Load and prepare data
data = load_iris()
X, y_raw = data.data, data.target

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)

ohe = OneHotEncoder()
y = ohe.fit_transform(y_raw.reshape(-1, 1))  # shape: (150, 3)

# Manual split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train
model = Sequential()
model.add(Dense(16, input_dim=4))
model.add(Relu())
model.add(Dense(8))
model.add(Relu())
model.add(Dense(3))

model.compile(loss=CCE(), optimizer=MomentumSGD(lr=0.01, beta=0.9))
model.fit(X_train, y_train, epochs=300, batch_size=16)

# Evaluate
probs = model.predict(X_test)
predictions = np.argmax(probs, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

---

### 4.2 Binary Classification with BCE Loss

```python
import numpy as np
from axiom import Sequential, Dense, Relu, BCE, Sigmoid
from axiom.core.optimizers import SGD

model = Sequential()
model.add(Dense(32, input_dim=20))
model.add(Relu())
model.add(Dense(16))
model.add(Relu())
model.add(Dense(1))
model.add(Sigmoid())

model.compile(loss=BCE(), optimizer=SGD(lr=0.05))
model.fit(X_train, y_train, epochs=500, batch_size=32)
```

---

### 4.3 Regression with Log-Cosh Loss

```python
import numpy as np
from axiom import Sequential, Dense, Relu, LOG_COSH
from axiom.core.optimizers import MomentumSGD

# No activation on the output — raw linear output for regression
model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Relu())
model.add(Dense(32))
model.add(Relu())
model.add(Dense(1))   # Single continuous output

model.compile(loss=LOG_COSH(), optimizer=MomentumSGD(lr=0.001, beta=0.9))
model.fit(X_train, y_train.reshape(-1, 1), epochs=1000, batch_size=64)
```

---

### 4.4 Decision Tree vs Logistic Regression

```python
from axiom import DecisionTreeClassifier, LogisticRegression
from axiom.preprocessing.scalers import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
tree.fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)

# Logistic Regression
lr = LogisticRegression(learning_rate=0.1, epochs=1000)
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)

print(f"Decision Tree Accuracy:   {tree_acc:.4f}")
print(f"Logistic Regression Acc:  {lr_acc:.4f}")
```

---

### 4.5 XOR Problem (Classic MLP Test)

The XOR function is not linearly separable — it is the canonical test that a single-layer network cannot solve but a two-layer MLP can.

```python
import numpy as np
from axiom import Sequential, Dense, Relu, Sigmoid, BCE
from axiom.core.optimizers import MomentumSGD

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0],   [1],   [1],   [0]],   dtype=float)

model = Sequential()
model.add(Dense(4, input_dim=2))
model.add(Relu())
model.add(Dense(1))
model.add(Sigmoid())

model.compile(loss=BCE(), optimizer=MomentumSGD(lr=0.1, beta=0.9))
model.fit(X, y, epochs=2000, batch_size=4)

print(model.predict(X).round(2))
# Expected: [[0.], [1.], [1.], [0.]]
```

---

## 5. Extending Axiom

Axiom is designed to be extended. New layers, activations, and losses can be added by inheriting from the base classes in `axiom/core/base.py`.

### Custom Layer

```python
from axiom.core.base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    """Randomly zeroes units during training to prevent co-adaptation."""

    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, A):
        if self.training:
            # Scale kept units by 1/(1-rate) to preserve expected activation magnitude
            self.mask = (np.random.rand(*A.shape) > self.rate) / (1 - self.rate)
            return A * self.mask
        return A  # No dropout at inference time

    def backward(self, dA):
        return dA * self.mask  # Gradient flows only through kept units
```

### Custom Loss Function

```python
from axiom.core.base import BaseLoss
import numpy as np

class HuberLoss(BaseLoss):
    """Less sensitive to outliers than MSE for regression tasks."""

    def __init__(self, delta=1.0):
        self.delta = delta

    def cost(self, y_pred, y_true):
        residual = np.abs(y_pred - y_true)
        return np.mean(np.where(
            residual <= self.delta,
            0.5 * residual**2,
            self.delta * residual - 0.5 * self.delta**2
        ))

    def gradient(self, y_pred, y_true):
        residual = y_pred - y_true
        return np.where(
            np.abs(residual) <= self.delta,
            residual,
            self.delta * np.sign(residual)
        ) / y_pred.shape[0]
```

### Custom Optimizer

```python
from axiom.core.base import BaseOptimizer
import numpy as np

class RMSProp(BaseOptimizer):
    """Adapts learning rates per-parameter using a running average of squared gradients."""

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}

    def update(self, layer, layer_id):
        if layer_id not in self.s:
            self.s[layer_id] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

        self.s[layer_id]["W"] = self.beta * self.s[layer_id]["W"] + (1 - self.beta) * layer.dW**2
        self.s[layer_id]["b"] = self.beta * self.s[layer_id]["b"] + (1 - self.beta) * layer.db**2

        layer.W -= self.lr * layer.dW / (np.sqrt(self.s[layer_id]["W"]) + self.epsilon)
        layer.b -= self.lr * layer.db / (np.sqrt(self.s[layer_id]["b"]) + self.epsilon)
```

Once defined, custom components plug directly into `Sequential` or `Model` without any other changes:

```python
model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Relu())
model.add(Dropout(rate=0.3))
model.add(Dense(1))
model.compile(loss=HuberLoss(delta=1.5), optimizer=RMSProp(lr=0.001))
```

---

_Documentation maintained alongside `axiom-ml` v0.1.0. For issues or suggestions, open a GitHub issue._
