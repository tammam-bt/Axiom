# Axiom (`axiom-ml`) — Technical Documentation

> **Version:** 0.1.2 | **Author:** Tammam BenBettayeb | **Python:** 3.9+ | **License:** MIT | **Dependency:** NumPy only

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

Every layer in Axiom implements a `backward(output_gradient, learning_rate)` method that carries out this computation, using the input cached during the forward pass. `Dense` also applies its own weight update inside `backward()`.

### 2.3 Weight Initialization

Initializing all weights to zero causes **symmetry breaking failure** — every neuron learns the same thing. Initializing with naively large values causes **exploding gradients**. Axiom implements three principled strategies (`Sequential` picks one for each `Dense` layer from the activation that follows it):

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

Because ReLU kills half of its inputs (the negative half), He initialization compensates by using a larger variance: `Var(W) = 2 / n_in`. This is also the correct choice for **Leaky ReLU**, which is a ReLU variant.

**LeCun Initialization** (used for SELU):

```
W ~ Uniform(-√(3 / n_in),  +√(3 / n_in))
```

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

In `Dense` the penalty is applied directly in the weight update, with `λ` passed as `l2_lambda` (default `0.0`, no regularization):

```
W := W - lr * (v_W + λ * W)
```

This extra term `λ * W` effectively **shrinks the weights at every step** (hence "weight decay"), making it harder for the network to memorize and overfit the training data.

---

## 3. API Reference

All classes below are accessible directly from the top-level `axiom` package. Deep imports from submodules also work and are shown alongside each class.

### 3.1 Model & Sequential

```python
from axiom import Sequential, Model
# or: from axiom.neural.engine import Sequential, Model
```

`Sequential` holds the stack of layers and runs the forward and backward passes through it. `Model` wraps a `Sequential` with a loss and the training loop. The loss is chosen by name when the `Model` is built, and the learning rate is passed to `fit`.

#### `Sequential`

```python
import numpy as np
from axiom import Sequential, Dense, Relu

net = Sequential([
    Dense(784, 64),
    Relu(),
    Dense(64, 10),
])
net.add(Relu())                       # layers can also be appended one at a time

out = net.forward(np.zeros((2, 784)))
print(out.shape)                      # (2, 10)
```

#### `Model`

```python
import numpy as np
from axiom import Sequential, Model, Dense, Relu

X_train = np.random.default_rng(0).normal(size=(100, 8))
y_train = X_train.sum(axis=1, keepdims=True)

net = Sequential([Dense(8, 32), Relu(), Dense(32, 1)])
model = Model(net, loss="mse")

model.fit(X_train, y_train, epochs=200, lr=0.05)
predictions = model.predict(X_train)
```

`loss` is a lowercase name: `"mse"` (also the fallback for an unknown name), `"bce"`, `"cce"` or `"logcosh"`. The loss functions themselves are in [3.4](#34-loss-functions).

#### Methods

| Method                                  | Description                                                       |
| --------------------------------------- | ----------------------------------------------------------------- |
| `Sequential(layers=None)`               | Build a stack from a list of layers                               |
| `net.add(layer)`                        | Append a layer or activation to the stack                         |
| `net.forward(X)`                        | Run `X` through every layer                                       |
| `net.backward(grad, lr)`                | Backpropagate a loss gradient and update the `Dense` weights      |
| `Model(net, loss="MSE")`                | Wrap a `Sequential` with a loss, chosen by name                   |
| `model.fit(X, y, epochs, lr)`           | Full-batch training; prints the error about 100 times per run     |
| `model.predict(X)`                      | Forward pass; returns the raw output of the final layer           |

> **Note:** each `Dense` layer takes its input size explicitly: `Dense(input_size, output_size)`. There is no `compile` step, no mini-batching and no `evaluate`; compute a score yourself from `model.predict`.

---

### 3.2 Layers

```python
from axiom import Dense
# or: from axiom.neural.layers import Dense
```

#### `Dense(input_size, output_size, initialization="xavier", momentum_beta=0.9, l2_lambda=0.0)`

A fully connected layer. Computes `Z = XW + b`.

| Parameter        | Type    | Default    | Description                                                            |
| ---------------- | ------- | ---------- | ---------------------------------------------------------------------- |
| `input_size`     | `int`   | —          | Number of inputs                                                       |
| `output_size`    | `int`   | —          | Number of neurons (output dimension)                                   |
| `initialization` | `str`   | `"xavier"` | Weight init: `"xavier"`, `"he"` (or `"kaiming"`), `"lecun"`            |
| `momentum_beta`  | `float` | `0.9`      | Momentum coefficient for the layer's own update                        |
| `l2_lambda`      | `float` | `0.0`      | L2 penalty strength                                                    |

Inside a `Sequential`, the initialization is replaced by the one that suits the activation that follows the layer (see [2.3](#23-weight-initialization)).

**Attributes:**

- `layer.weights` — weight matrix, shape `(input_size, output_size)`
- `layer.bias` — bias vector, shape `(1, output_size)`
- `layer.velocity` — momentum buffer, same shape as `weights`
- `layer.input` — the input cached by the last `forward()`

**Core methods:**

- `layer.forward(input_data)` → `Z` — linear transform, caches the input
- `layer.backward(output_gradient, learning_rate)` → gradient for the previous layer — updates `weights` and `bias` in place and returns the upstream gradient

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

Each loss is a plain function `loss(y_true, y_predicted)` that returns a scalar. The matching gradient function (`MSE_derivative`, `BCE_derivative`, `CCE_derivative`, `LOG_COSH_derivative`, same arguments) lives in `axiom.core.losses`. `Model` picks the pair by name.

```python
import numpy as np
from axiom import MSE, BCE

y_true = np.array([[1.0], [0.0]])
y_pred = np.array([[0.9], [0.2]])
print(MSE(y_true, y_pred), BCE(y_true, y_pred))
```

#### `MSE` — Mean Squared Error

```
J = mean((ŷ - y)²)
dA = (2/m) * (ŷ - y)
```

Use for **regression** tasks where the output is a continuous value.

#### `BCE` — Binary Cross-Entropy

```
J = -mean[ y·log(ŷ) + (1-y)·log(1-ŷ) ]
dA = (1/m) * (ŷ - y) / (ŷ(1-ŷ))
```

Use for **binary classification** (a sigmoid output). When the last layer is a `Sigmoid` and the loss is `"bce"`, `Model` uses the fused gradient `(ŷ - y) / m` and skips the sigmoid's own derivative.

#### `CCE` — Categorical Cross-Entropy

```
J = -mean[ y·log(ŷ) ]
dA = -(y/ŷ) / m
```

Takes one-hot targets and predicted probabilities. Axiom has no softmax layer yet, so the predictions have to come from your own normalisation; the fused softmax gradient is not implemented. For several classes with the layers that exist today, a sigmoid output per class with `"bce"` is the simplest route (see [4.1](#41-multi-class-classification-one-vs-rest)).

#### `LOG_COSH` — Log-Cosh Loss

```
J = mean[ log(cosh(ŷ - y)) ]
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

`momentum=` is accepted as another name for `beta`. Both optimizers update a list of arrays in place with `opt.step(params, grads)`, using the rules in [2.4](#24-optimization). They are standalone for now: `Dense` still applies its own momentum update inside `backward()`.

```python
import numpy as np
from axiom.core.optimizers import MomentumSGD

w = np.array([1.0, -2.0])
opt = MomentumSGD(lr=0.1, beta=0.9)
for _ in range(100):
    opt.step([w], [2 * w])      # gradient of sum(w**2)
print(w)                        # close to [0, 0]
```

---

### 3.6 Classical Algorithms

#### Linear Regression

```python
import numpy as np
from axiom import LinearRegression, MSE
# or: from axiom.linear.linear_regression import LinearRegression

rng = np.random.default_rng(0)
X_train, X_test = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
y_train = X_train @ np.array([1.5, -2.0, 0.5]) + 3.0
y_test = X_test @ np.array([1.5, -2.0, 0.5]) + 3.0

model = LinearRegression(alpha=0.1)          # alpha is the learning rate
model.fit(X_train, y_train, epochs=1000)
predictions = model.predict(X_test)
mse = MSE(y_test, predictions)
```

`fit(x, y, epochs=1000, with_logging=False)` returns the cost after each epoch when `with_logging=True`, otherwise `None`. The fitted parameters are `model.weights` and `model.bias`.

#### Logistic Regression

```python
import numpy as np
from axiom import LogisticRegression
# or: from axiom.linear.logistic_regression import LogisticRegression

rng = np.random.default_rng(0)
X_train, X_test = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

model = LogisticRegression(alpha=0.1)         # alpha is the learning rate
model.fit(X_train, y_train, epochs=500)
preds = model.predict(X_test)                 # Returns 0/1 class labels
probs = model.predict_proba(X_test)           # Returns sigmoid probabilities
acc = np.mean(preds == y_test)
```

`fit(X, y, epochs=1000, with_logging=False)` starts from the same point every time (weights of one, bias from the constructor's `b`), so fitting twice on the same data gives the same parameters. The fitted parameters are `model.w` and `model.b`.

#### Decision Trees

```python
from axiom import DecisionTreeClassifier, DecisionTreeRegressor
# or: from axiom.trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

import numpy as np
rng = np.random.default_rng(0)
X_train, X_test = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
y_train = (X_train[:, 0] > 0).astype(int)

tree = DecisionTreeClassifier(criterion="gini", max_depth=10)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)

reg = DecisionTreeRegressor(max_depth=4)
reg.fit(X_train, X_train[:, 0] * 2.0)
values = reg.predict(X_test)
```

`criterion` options: `"entropy"` (information gain, the default) or `"gini"` (Gini impurity); anything else raises `ValueError`. Both are computed from class counts, so any number of classes and any label type works. `DecisionTreeRegressor` splits on variance and its leaves predict the mean. `max_depth` counts the root as depth 0 (`None` means no limit), and `tree.get_depth()` returns the fitted depth. The candidate thresholds are the midpoints between consecutive distinct values of each feature. `gini`, `entropy`, `variance` and `information_gain(y, left_mask, impurity)` live in `axiom/trees/gains.py`, next to the count-based versions the tree uses to score every threshold at once.

---

### 3.7 Tree Ensemble Methods

All ensemble methods live in `axiom/trees/` and are built on top of the same decision tree primitives.

#### Gradient Boosting Machine (GBM)

```python
from axiom.trees.gbm import GBM, GBMRegressor, GBMClassifier   # GBM is GBMRegressor

import numpy as np
rng = np.random.default_rng(0)
X_train, X_test = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
y_train = X_train[:, 0] * 2.0
y_binary = (X_train[:, 0] > 0).astype(int)

model = GBM(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)

clf = GBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X_train, y_binary)           # exactly two classes
labels = clf.predict(X_test)
probs = clf.predict_proba(X_test)    # probability of clf.classes_[1]
```

GBM fits an additive sequence of weak learners (shallow trees), where each new tree is trained on the **pseudo-residuals** (negative gradients of the loss) from the current ensemble. `GBMRegressor` uses squared error and starts from the mean of `y`. `GBMClassifier` uses log-loss, starts from the log-odds of the positive class and fits each tree to `y - sigmoid(F)`; each leaf then takes one Newton step, `sum(y - s) / sum(s(1 - s))` with `s = sigmoid(F)`. `model.train_loss_` holds the training loss before the first tree and after each tree.

#### XGBoost

> **In progress.** `axiom/trees/xgboost.py` is still empty, so `XGBoost` cannot be imported yet. The API below is the plan.

```python
# planned API, not importable yet
from axiom.trees.xgboost import XGBoost

model = XGBoost(n_estimators=100, learning_rate=0.1, max_depth=4, lambda_=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

Extends GBM by using **second-order Taylor expansions** of the loss for more accurate gradient steps, and adds L2 regularization (`lambda_`) directly on the leaf weights.

#### LightGBM

> **In progress.** `axiom/trees/lightgbm.py` is still empty, so `LightGBM` cannot be imported yet. The API below is the plan.

```python
# planned API, not importable yet
from axiom.trees.lightgbm import LightGBM

model = LightGBM(n_estimators=100, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

Uses **histogram-based splitting** and **leaf-wise (best-first) tree growth** rather than depth-wise growth, which reduces memory and speeds up training on larger datasets.

#### Linear Tree

> **In progress.** `axiom/trees/linear_tree.py` is still empty, so `LinearTree` cannot be imported yet. The API below is the plan.

```python
# planned API, not importable yet
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
import numpy as np
from axiom.preprocessing.scalers import MinMaxScaler, StandardScaler

X = np.random.default_rng(0).normal(size=(100, 3))
X_train, X_test = X[:80], X[80:]

scaler = StandardScaler()          # Zero mean, unit variance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # Uses stats from training set only

scaler = MinMaxScaler()            # Scales to [0, 1]
X_scaled = scaler.fit_transform(X)
```

> **Important:** Always call `fit_transform` on training data and `transform` (never `fit_transform`) on test/validation data. Fitting on test data leaks information and inflates performance metrics.

#### Encoders

```python
from axiom.preprocessing.encoders import LabelEncoder, OneHotEncoder

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

The examples build their own small synthetic data, so each one runs as it stands. Swap in your own arrays of shape `(n_samples, n_features)`.

### 4.1 Multi-Class Classification (one-vs-rest)

Axiom has no softmax layer yet, so the network gets one sigmoid output per class and is trained with `"bce"` against one-hot targets.

```python
import numpy as np
from axiom import Sequential, Model, Dense, Relu, Sigmoid
from axiom.preprocessing.scalers import StandardScaler
from axiom.preprocessing.encoders import OneHotEncoder

# Three Gaussian blobs in four dimensions
rng = np.random.default_rng(0)
centres = np.array([[0, 0, 0, 0], [3, 3, 0, 0], [0, 3, 3, 3]], dtype=float)
y_raw = rng.integers(0, 3, size=150)
X = centres[y_raw] + rng.normal(size=(150, 4))

# Preprocess
X = StandardScaler().fit_transform(X)
y = OneHotEncoder().fit_transform(y_raw)  # shape: (150, 3)

# Manual split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train
net = Sequential([
    Dense(4, 16), Relu(),
    Dense(16, 8), Relu(),
    Dense(8, 3), Sigmoid(),
])
model = Model(net, loss="bce")
model.fit(X_train, y_train, epochs=300, lr=0.1)

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
from axiom import Sequential, Model, Dense, Relu, Sigmoid

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 20))
y_train = (X_train[:, :5].sum(axis=1) > 0).astype(float).reshape(-1, 1)

net = Sequential([
    Dense(20, 32), Relu(),
    Dense(32, 16), Relu(),
    Dense(16, 1), Sigmoid(),
])
model = Model(net, loss="bce")
model.fit(X_train, y_train, epochs=500, lr=0.05)
```

---

### 4.3 Regression with Log-Cosh Loss

```python
import numpy as np
from axiom import Sequential, Model, Dense, Relu

rng = np.random.default_rng(0)
X_train = rng.normal(size=(200, 10))
y_train = X_train[:, :3].sum(axis=1)

# No activation on the output — raw linear output for regression
net = Sequential([
    Dense(10, 64), Relu(),
    Dense(64, 32), Relu(),
    Dense(32, 1),          # Single continuous output
])
model = Model(net, loss="logcosh")
model.fit(X_train, y_train.reshape(-1, 1), epochs=1000, lr=0.01)
```

---

### 4.4 Decision Tree vs Logistic Regression

```python
import numpy as np
from axiom import DecisionTreeClassifier, LogisticRegression
from axiom.preprocessing.scalers import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

X_scaled = StandardScaler().fit_transform(X)

split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, criterion="entropy")
tree.fit(X_train, y_train)
tree_acc = np.mean(tree.predict(X_test) == y_test)

# Logistic Regression
lr = LogisticRegression(alpha=0.1)
lr.fit(X_train, y_train, epochs=1000)
lr_acc = np.mean(lr.predict(X_test) == y_test)

print(f"Decision Tree Accuracy:   {tree_acc:.4f}")
print(f"Logistic Regression Acc:  {lr_acc:.4f}")
```

---

### 4.5 XOR Problem (Classic MLP Test)

The XOR function is not linearly separable — it is the canonical test that a single-layer network cannot solve but a two-layer MLP can.

```python
import numpy as np
from axiom import Sequential, Model, Dense, Tanh, Sigmoid

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0],   [1],   [1],   [0]],   dtype=float)

net = Sequential([
    Dense(2, 8), Tanh(),
    Dense(8, 1), Sigmoid(),
])
model = Model(net, loss="bce")
model.fit(X, y, epochs=5000, lr=0.5)

print(model.predict(X).round(2))
# Typically close to [[0.], [1.], [1.], [0.]]; the result depends on the random start.
```

---

## 5. Extending Axiom

New layers can be added by inheriting from `Layer` in `axiom/neural/layers.py` and plugged into a `Sequential`. Losses and optimizers are plain functions and small classes, shown below with the interface they follow.

### Custom Layer

A layer implements `forward(input_data)` and `backward(output_gradient, learning_rate)`.

```python
import numpy as np
from axiom import Sequential, Model, Dense, Relu
from axiom.neural.layers import Layer

class Dropout(Layer):
    """Randomly zeroes units during training to prevent co-adaptation."""

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, input_data):
        if self.training:
            # Scale kept units by 1/(1-rate) to preserve expected activation magnitude
            self.mask = (np.random.rand(*input_data.shape) > self.rate) / (1 - self.rate)
            return input_data * self.mask
        return input_data  # No dropout at inference time

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask  # Gradient flows only through kept units
```

It plugs directly into a `Sequential`. Set `training = False` before predicting. Run this together with the class above:

```python
# continues the Dropout example above
rng = np.random.default_rng(0)
X_train = rng.normal(size=(100, 10))
y_train = X_train[:, :2].sum(axis=1, keepdims=True)

dropout = Dropout(rate=0.3)
net = Sequential([Dense(10, 64), Relu(), dropout, Dense(64, 1)])
model = Model(net, loss="mse")
model.fit(X_train, y_train, epochs=100, lr=0.01)

dropout.training = False
predictions = model.predict(X_train)
```

### Custom Loss Function

A loss is a function of `(y_true, y_predicted)` and a matching derivative, like the ones in `axiom/core/losses.py`. `Model` looks losses up by name in `_choose_loss`, so to train with a new one, add a branch there; until then call the pair yourself.

```python
import numpy as np

def huber(y_true, y_predicted, delta=1.0):
    """Less sensitive to outliers than MSE for regression tasks."""
    residual = np.abs(y_predicted - y_true)
    return np.mean(np.where(
        residual <= delta,
        0.5 * residual**2,
        delta * residual - 0.5 * delta**2
    ))

def huber_derivative(y_true, y_predicted, delta=1.0):
    residual = y_predicted - y_true
    return np.where(
        np.abs(residual) <= delta,
        residual,
        delta * np.sign(residual)
    ) / y_true.shape[0]

y_true = np.array([[0.0], [0.0]])
y_pred = np.array([[0.5], [3.0]])
print(huber(y_true, y_pred), huber_derivative(y_true, y_pred))
```

### Custom Optimizer

An optimizer follows the interface of `SGD` and `MomentumSGD`: `step(params, grads)` updates a list of NumPy arrays in place.

```python
import numpy as np

class RMSProp:
    """Adapts learning rates per-parameter using a running average of squared gradients."""

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def step(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(p, dtype=float) for p in params]
        for p, g, s in zip(params, grads, self.s):
            s *= self.beta
            s += (1 - self.beta) * g**2
            p -= self.lr * g / (np.sqrt(s) + self.epsilon)

w = np.array([1.0, -2.0])
opt = RMSProp(lr=0.05)
for _ in range(200):
    opt.step([w], [2 * w])      # gradient of sum(w**2)
print(w)
```

---

_Documentation maintained alongside `axiom-ml` v0.1.2. For issues or suggestions, open a GitHub issue._
