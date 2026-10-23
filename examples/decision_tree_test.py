

import numpy as np

from axiom.trees.decision_tree import DecisionTreeClassifier


print("Initializing sanity check...")

# 1. More complex dataset (XOR pattern)
# This cannot be separated with a single split; it needs depth >= 2.
# Rule: class 1 if exactly one feature is > 0.5, else class 0.
X = np.array([
    # Quadrant A: x0 < 0.5, x1 < 0.5  -> class 0
    [0.10, 0.12], [0.18, 0.20], [0.22, 0.28], [0.30, 0.35],
    [0.40, 0.18], [0.12, 0.42], [0.33, 0.10], [0.45, 0.45],

    # Quadrant B: x0 < 0.5, x1 > 0.5  -> class 1
    [0.08, 0.62], [0.15, 0.75], [0.25, 0.88], [0.35, 0.70],
    [0.42, 0.58], [0.20, 0.95], [0.47, 0.82], [0.05, 0.55],

    # Quadrant C: x0 > 0.5, x1 < 0.5  -> class 1
    [0.55, 0.05], [0.62, 0.22], [0.70, 0.40], [0.82, 0.12],
    [0.95, 0.30], [0.78, 0.48], [0.58, 0.35], [0.88, 0.18],

    # Quadrant D: x0 > 0.5, x1 > 0.5  -> class 0
    [0.55, 0.60], [0.63, 0.72], [0.74, 0.84], [0.86, 0.66],
    [0.92, 0.94], [0.68, 0.55], [0.97, 0.58], [0.79, 0.90]
])

y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,   # Quadrant A
    1, 1, 1, 1, 1, 1, 1, 1,   # Quadrant B
    1, 1, 1, 1, 1, 1, 1, 1,   # Quadrant C
    0, 0, 0, 0, 0, 0, 0, 0    # Quadrant D
])

print("Dataset created. Starting tree training...")

# 2. Initialize and train the tree
# Setting max_depth=1 to strictly test the first calculation
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, verbose=False)

print(tree.compute_information_gain(X, y, threshold=0.2, feature=0)["info_gain"])  # Debug statement to check info gain calculation for feature 0
print(tree.compute_information_gain(X, y, threshold=0.7, feature=1)["info_gain"])  # Debug statement to check info gain calculation for feature 1

tree.fit(X, y)
tree.print_tree()
x_predict = np.array([[0.2, 0.1], [0.2, 0.8], [0.7, 0.4]])
predictions = tree.predict(x_predict)
print("Predictions for test inputs:", predictions)