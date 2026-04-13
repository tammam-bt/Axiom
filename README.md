<img src="https://cdn-icons-png.flaticon.com/512/7747/7747363.png" alt="Axiom Library Logo" height="150">

# Axiom (`axiom-ml`)

<p align="center">
  <img src="https://img.shields.io/pypi/v/axiom-ml?color=blue&label=PyPI&logo=pypi&logoColor=white" alt="PyPI Version"/>
  <img src="https://img.shields.io/pypi/pyversions/axiom-ml?logo=python&logoColor=white" alt="Python Versions"/>
  <img src="https://img.shields.io/github/license/tammam-bt/axiom?color=green" alt="License"/>
  <img src="https://img.shields.io/pypi/dm/axiom?color=orange&label=Downloads" alt="Downloads"/>
  <img src="https://img.shields.io/badge/dependencies-NumPy%20only-lightgrey" alt="Dependencies"/>
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status"/>
</p>

> **A from-scratch, NumPy-only machine learning library built to make the math visible.**

Axiom is a lightweight, educational ML library that implements deep learning and classical algorithms from first principles — no PyTorch, no TensorFlow, no autograd magic. Every forward pass, every gradient, every weight update is written explicitly so you can read, trace, and understand exactly what is happening at each step.

Built as a learning project during the first year of CS engineering at [ENSI (École Nationale des Sciences de l'Informatique)](https://ensi.rnu.tn/), Axiom is both a proof-of-concept and a functional prototyping tool for students and developers who want to go beyond calling `.fit()`.

---

## Why Axiom?

Most ML frameworks abstract away the calculus. That is great for production — but terrible for learning. Axiom takes the opposite bet: **every operation is transparent, every derivative is hand-written, and every design choice is documented.**

| Feature | Axiom | Scikit-learn | PyTorch |
|---|---|---|---|
| Dependencies | NumPy only | Many | Many |
| Manual backprop | ✅ | ❌ | ❌ (autograd) |
| Readable internals | ✅ | Partial | ❌ |
| Educational focus | ✅ | Partial | ❌ |
| Production-ready | ❌ (by design) | ✅ | ✅ |

---

## Feature Overview

### Classical Supervised Learning
- **Linear Regression** — gradient descent implementation
- **Logistic Regression** — binary classification with sigmoid output
- **Decision Tree Classifier** — recursive splitting with information gain / Gini impurity
- **Gradient Boosting Machines (GBM)** — ensemble boosting over decision stumps
- **XGBoost** — second-order gradient boosting with regularization
- **LightGBM** — histogram-based, leaf-wise tree growth
- **Linear Tree** — hybrid model combining linear functions at leaf nodes

### Deep Learning
- **Sequential / Model API** — two interfaces for building and training neural networks
- **Dense (Fully Connected) Layers** — with weight/bias state management
- **Activations** — ReLU, Leaky ReLU, Sigmoid, Tanh, SELU (forward + analytical derivative for each)
- **Loss Functions** — MSE, Binary Cross-Entropy (BCE), Categorical Cross-Entropy (CCE), Log-Cosh
- **Optimizers** — Vanilla SGD, Momentum SGD

### Preprocessing
- **Scalers** — feature normalization and standardization utilities
- **Encoders** — label and one-hot encoding utilities

---

## Installation

Axiom requires Python 3.8+ and NumPy. Install via pip:

```bash
pip install axiom-ml
```

Or clone and install locally for development:

```bash
git clone https://github.com/your-username/axiom-ml.git
cd axiom-ml
pip install -e .
```

Verify the installation:

```python
import axiom
axiom.info()
# Axiom ML Library v0.1.0
# Status: Operational. Optimized for NumPy-based learning.
```

---

## Quickstart

### Train a Neural Network

```python
import numpy as np
from axiom import Sequential, Dense, Relu, Model

# Build the model using the Sequential API
Sequential = Sequential([
Dense(728,64),
Relu(),
Dense(64,32),
Relu(),
Dense(32,1),
])

model = Model(Sequential, loss="MSE")
model.fit(X_train, y_train, epochs=50, lr=0.1)

predictions = model.predict(X_test)
```

### Binary Classification (Logistic Regression)

```python
from axiom import LogisticRegression

clf = LogisticRegression(learning_rate=0.1, epochs=1000)
clf.fit(X_train, y_train)

Predictions = clf.predict(X_test, y_test)
print(f"Predictions: {Predictions}")
```

### Decision Tree

```python
from axiom import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, criterion="gini")
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
```

---

## Project Structure

```
axiom-repo/
├── src/
│   └── axiom/
│       ├── __init__.py          # Public API — all top-level imports live here
│       ├── core/
│       │   ├── base.py          # Abstract base classes (BaseLayer, BaseLoss, ...)
│       │   ├── losses.py        # MSE, BCE, CCE, LOG_COSH
│       │   └── optimizers.py    # SGD, MomentumSGD
│       ├── neural/
│       │   ├── engine.py        # Sequential and Model training engines
│       │   ├── layers.py        # Dense layer
│       │   └── activations.py   # Relu, LeakyRelu, Sigmoid, Tanh, Selu
│       ├── linear/
│       │   ├── linear_regression.py
│       │   └── logistic_regression.py
│       ├── trees/
│       │   ├── decision_tree.py # DecisionTreeClassifier
│       │   ├── gains.py         # Gini impurity, entropy, information gain
│       │   ├── gbm.py           # Gradient Boosting Machines
│       │   ├── xgboost.py       # XGBoost implementation
│       │   ├── lightgbm.py      # LightGBM implementation
│       │   └── linear_tree.py   # Linear model at leaf nodes
│       └── preprocessing/
│           ├── scalers.py       # Feature scaling utilities
│           └── encoders.py      # Label and one-hot encoders
├── tests/
│   ├── decision_tree_test.py
│   ├── linear_regression_test.py
│   ├── logistic_regression_test.py
│   ├── refactoring_test.py
│   └── test_xor.py
├── docs/
│   └── manual.md
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Design Philosophy

Axiom is built around three convictions:

1. **Transparency over convenience.** No hidden operations. If a weight is updated, you can find exactly where and why in the source.
2. **Math-first implementation.** Every algorithm maps directly to its mathematical definition. Reading the code and reading the equations should feel equivalent.
3. **Modular by design.** Components (layers, activations, losses, optimizers) are fully decoupled. Swap any piece without touching the rest.

---

## Contributing

Contributions, issues, and suggestions are welcome! This project is especially friendly to fellow students working through ML fundamentals.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please make sure any new algorithm includes its mathematical derivation in the docstring.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

Built from scratch during Year 1 at **ENSI — École Nationale des Sciences de l'Informatique**, Tunisia.  
By **Tammam BenBettayeb** — inspired by the "implement it yourself to understand it" philosophy of foundational CS education.

📜 Licensing
------------

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

***
