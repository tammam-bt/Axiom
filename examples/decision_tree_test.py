

import numpy as np

from axiom.trees.decision_tree import DecisionTreeClassifier
from axiom.trees.gains import entropy, information_gain


print("Initializing sanity check...")

# 1. XOR dataset: the four corners of the unit square, eight copies each (32 points).
# Rule: class 1 if exactly one feature is 1, else class 0.
# No single split separates it, so the tree needs depth 2.
corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
X = np.tile(corners, (8, 1))
y = np.tile([0, 1, 1, 0], 8)

print("Dataset created. Starting tree training...")

# 2. Initialize and train the tree (max_depth=3 is enough; it stops at depth 2)
tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, verbose=False)

# Either first split on its own gains nothing: each side still holds half 0s and half 1s.
# The tree takes it anyway (the best gain is 0, not negative), and the second level separates the classes.
print("Gain of x0 <= 0.5:", information_gain(y, X[:, 0] <= 0.5, entropy))
print("Gain of x1 <= 0.5:", information_gain(y, X[:, 1] <= 0.5, entropy))

tree.fit(X, y)
tree.print_tree()
x_predict = np.array([[0.2, 0.1], [0.2, 0.8], [0.7, 0.4], [0.9, 0.7]])
predictions = tree.predict(x_predict)
print("Predictions for test inputs:", predictions)  # expected [0 1 1 0]
print("Training accuracy:", np.mean(tree.predict(X) == y))