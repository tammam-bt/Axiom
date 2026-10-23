"""
Axiom: A NumPy-based educational library for Neural Networks and Decision Trees.
Built by a freshman CS Engineering student at ENSI.
"""

__version__ = "0.1.2"
__author__ = "Tammam BenBettayeb"

# A failed import here should stop `import axiom` loudly, not print a warning
# and leave half of __all__ undefined.
from .trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .trees.gbm import GBM, GBMClassifier, GBMRegressor
from .neural.engine import Sequential, Model
from .neural.layers import Dense
from .neural.activations import LeakyRelu, Relu, Sigmoid, Tanh, Selu
from .core.losses import MSE, BCE, CCE, LOG_COSH
from .core.optimizers import SGD, MomentumSGD
from .preprocessing.encoders import LabelEncoder, OneHotEncoder
from .linear.linear_regression import LinearRegression
from .linear.logistic_regression import LogisticRegression

__all__ = [
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "GBM", "GBMClassifier", "GBMRegressor",
    "Sequential", "Model",
    "Dense", "LeakyRelu", "Relu", "Sigmoid", "Tanh", "Selu",
    "MSE", "BCE", "CCE", "LOG_COSH",
    "SGD", "MomentumSGD",
    "LabelEncoder", "OneHotEncoder",
    "LinearRegression", "LogisticRegression",
    "__version__"
]

import numpy as np
np.set_printoptions(precision=4, suppress=True)

def info():
    """Utility function to check library status."""
    print(f"Axiom ML Library v{__version__}")
    print("Status: Operational. Optimized for NumPy-based learning.")