import numpy as np
import axiom.neural.engine as nn

def test_linear_convergence():
    print("=== Axiom Linear Consistency Test ===")
    
    # 1. Generate Synthetic Linear Data: y = 2x1 + 3x2 + 5
    np.random.seed(42)
    X = np.random.rand(100, 2)
    true_weights = np.array([[2], [3]])
    true_bias = 5
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, 0.01, (100, 1))

    # 2. Calculate Ground Truth (Normal Equation)
    # Adding a column of ones for the bias term
    X_b = np.c_[np.ones((100, 1)), X] 
    best_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(f"Mathematical Ground Truth: Bias={best_theta[0][0]:.4f}, W1={best_theta[1][0]:.4f}, W2={best_theta[2][0]:.4f}")

    # 3. Setup Axiom NN (Purely Linear)
    # We use a single Dense layer with NO activation (Identity)
    network = nn.Sequential([
        nn.Dense(2, 1, momentum_beta=0) # Pure SGD for direct comparison
    ])
    
    model = nn.Model(network, loss="MSE")

    # 4. Train
    print("Training Axiom Linear Layer...")
    model.fit(X, y, epochs=1000, lr=0.1)

    # 5. Extract Axiom Results
    # Assuming your Dense layer stores weights in .W and bias in .b
    axiom_w = network.layers[0].weights
    axiom_b = network.layers[0].bias

    print(f"Axiom Results: Bias={axiom_b[0][0]:.4f}, W1={axiom_w[0][0]:.4f}, W2={axiom_w[1][0]:.4f}")

    # 6. Verification Logic
    mse_dist = np.mean((best_theta[1:] - axiom_w)**2)
    
    if mse_dist < 1e-3:
        print("\n✅ TEST PASSED: Axiom matches mathematical linear regression.")
    else:
        print("\n❌ TEST FAILED: Significant divergence from expected linear weights.")

if __name__ == "__main__":
    test_linear_convergence()