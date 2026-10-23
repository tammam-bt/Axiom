<img src="https://cdn-icons-png.flaticon.com/512/7747/7747363.png" alt="Axiom Library Logo" height="150">

# Axiom (`axiom-ml`)

<p align="center">
  <img src="https://img.shields.io/pypi/v/axiom-ml?color=blue&label=PyPI&logo=pypi&logoColor=white" alt="PyPI Version"/>
  <img src="https://img.shields.io/pypi/pyversions/axiom-ml?logo=python&logoColor=white" alt="Python Versions"/>
  <img src="https://img.shields.io/github/license/tammam-bt/axiom?color=green" alt="License"/>
  <img src="https://img.shields.io/pypi/dm/axiom-ml?color=orange&label=Downloads" alt="Downloads"/>
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
- **Decision Trees** — `DecisionTreeClassifier` (entropy or Gini, any number of classes) and `DecisionTreeRegressor` (variance), with a vectorized split search
- **Gradient Boosting Machines (GBM)** — `GBMRegressor` (squared error) and `GBMClassifier` (binary, log-loss), boosting shallow regression trees
- **XGBoost-style boosting** — second-order gradient boosting with regularization *(in progress)*
- **LightGBM-style boosting** — histogram-based, leaf-wise tree growth *(in progress)*
- **Linear Tree** — hybrid model combining linear functions at leaf nodes *(in progress)*

### Deep Learning
- **Sequential / Model API** — two interfaces for building and training neural networks
- **Dense (Fully Connected) Layers** — with weight/bias state management
- **Activations** — ReLU, Leaky ReLU, Sigmoid, Tanh, SELU (forward + analytical derivative for each)
- **Loss Functions** — MSE, Binary Cross-Entropy (BCE), Categorical Cross-Entropy (CCE), Log-Cosh
- **Optimizers** — `SGD` and `MomentumSGD` with `step(params, grads)`; standalone for now, as `Dense` still applies its own momentum update

### Preprocessing
- **Scalers** — `MinMaxScaler` and `StandardScaler` (fit / transform / inverse_transform with per-feature training statistics), and the functions `min_max_scale`, `standard_scale` and `max_abs_scale`
- **Encoders** — `LabelEncoder` and `OneHotEncoder` with fit / transform / inverse_transform

---

## Installation

Axiom requires Python 3.9+ and NumPy 1.24+. Install via pip:

```bash
pip install axiom-ml
```

Or clone and install locally for development:

```bash
git clone https://github.com/tammam-bt/Axiom.git
cd Axiom
pip install -e .
```

Verify the installation:

```python
import axiom
axiom.info()
# Axiom ML Library v0.1.2
# Status: Operational. Optimized for NumPy-based learning.
```

---

## Quickstart

### Train a Neural Network

```python
import numpy as np
from axiom import Sequential, Dense, Relu, Model

# Build the network with the Sequential API
net = Sequential([
Dense(728,64),
Relu(),
Dense(64,32),
Relu(),
Dense(32,1),
])

model = Model(net, loss="MSE")
model.fit(X_train, y_train, epochs=50, lr=0.1)

predictions = model.predict(X_test)
```

### Binary Classification (Logistic Regression)

```python
from axiom import LogisticRegression

clf = LogisticRegression(alpha=0.1)
clf.fit(X_train, y_train, epochs=1000)

preds = clf.predict(X_test)        # 0/1 labels
probs = clf.predict_proba(X_test)  # probability of class 1
print(f"Predictions: {preds}")
```

### Decision Tree

```python
from axiom import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="gini", max_depth=5)
tree.fit(X_train, y_train)
preds = tree.predict(X_test)
```

### Gradient Boosting

```python
from axiom import GBMClassifier

gbm = GBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)
preds = gbm.predict(X_test)
```

---

## Project Structure

```
Axiom/
├── src/
│   └── axiom/
│       ├── __init__.py          # Public API — all top-level imports live here
│       ├── core/
│       │   ├── base.py          # BaseEstimator
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
│       │   ├── decision_tree.py # DecisionTreeClassifier, DecisionTreeRegressor
│       │   ├── gains.py         # Gini impurity, entropy, variance, information gain
│       │   ├── gbm.py           # GBMRegressor, GBMClassifier
│       │   ├── xgboost.py       # in progress (empty)
│       │   ├── lightgbm.py      # in progress (empty)
│       │   └── linear_tree.py   # in progress (empty)
│       └── preprocessing/
│           ├── scalers.py       # MinMaxScaler, StandardScaler, scaling functions
│           └── encoders.py      # LabelEncoder, OneHotEncoder
├── tests/                       # pytest suite: pip install -e ".[test]" && pytest
│   ├── test_trees.py
│   ├── test_gbm.py
│   ├── test_linear.py
│   ├── test_preprocessing.py
│   └── test_optimizers.py
├── examples/                    # runnable scripts that print results
├── docs/
│   └── documentation.md
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

