from axiom.linear.linear_regression import LinearRegression

import numpy as np

# 1. Generate Synthetic Data: y = 4 + 3x1 + 2x2 + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 2)
true_weights = np.array([3, 2])
true_intercept = 4
y = true_intercept + X.dot(true_weights) + np.random.randn(100)

model = LinearRegression()
model.fit(X,y)
print(model.predict(X))

# 3. Test: Compare recovered weights to true weights
print(f"Recovered Weights: {model.weights}, Intercept: {model.bias}")
print (f"True Weights: {true_weights}, Intercept: {true_intercept}")