"""
Axiom: A NumPy-based educational library for Neural Networks and Decision Trees.
Built by a freshman CS Engineering student at ENSI.
"""

__version__ = "0.1.0"
__author__ = "Tammam BenBettayeb"

try:
    from .trees.decision_tree import DecisionTreeClassifier
    from .neural.engine import Sequential, Model
    from .neural.layers import Dense
    from .neural.activations import LeakyRelu, Relu, Sigmoid, Tanh, Selu
    from .core.losses import MSE, BCE, CCE, LOG_COSH
    from .linear.linear_regression import LinearRegression
    from .linear.logistic_regression import LogisticRegression
except ImportError as e:
    print(f"Warning: Could not import core components of Axiom: {e}")

__all__ = [
    "DecisionTreeClassifier",
    "Sequential", "Model",
    "Dense", "LeakyRelu", "Relu", "Sigmoid", "Tanh", "Selu",
    "MSE", "BCE", "CCE", "LOG_COSH",
    "LinearRegression", "LogisticRegression",
    "__version__"
]

import numpy as np
np.set_printoptions(precision=4, suppress=True)

def info():
    """Utility function to check library status."""
    print(f"Axiom ML Library v{__version__}")
    print("Status: Operational. Optimized for NumPy-based learning.")