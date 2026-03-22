import NN
import numpy as np
np.random.seed(42)
if __name__ == "__main__":
    # Training data (4 samples, 2 features each)
    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])

    # Targets (4 samples, 1 output each)
    y_train = np.array([[0], [1], [1], [0]])
    sequential = NN.Sequential(
        NN.Dense(2,3),
        NN.Tanh(),
        NN.Dense(3,1),
        NN.Sigmoid()
    )
    Model = NN.Model(sequential)
    Model.fit(x_train,y_train,epochs = 10000, lr = 0.1)
    print(Model.predict([[0,0],[0,1],[1,0],[1,1]]))