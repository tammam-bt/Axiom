import numpy as np
from sklearn.datasets import make_blobs
from axiom.linear.logisitc_regression import LogisticRegression

# 1. Generate linearly separable clusters
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=0.5, random_state=42)
print(X)
print(y)

# 2. Train your Axiom/Custom Logistic Model
model = LogisticRegression()
model.fit(X, y, epochs=100)

# 3. Test: Accuracy should be 100% or very close for separable data
print(np.mean((model.predict(X)>0.1) == y)) # Should return 0 or 1
print(f"Sanity Check Accuracy: {model.cost_function(X,y)}")