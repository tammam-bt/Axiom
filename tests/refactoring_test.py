import numpy as np
import axiom.neural.engine as nn

import numpy as np
np.random.seed(42)
if __name__ == "__main__":
    x_train = np.array([
        [ 1,  1],
        [-1,  1],
        [-1, -1],
        [ 1, -1]
    ])

    y_train = np.array([[1], [0], [0], [0]])
    personal_sequential = nn.Sequential([
        nn.Dense(2,5),
        nn.LeakyRelu(),
        nn.Dense(5,5),
        nn.LeakyRelu(),
        nn.Dense(5,1),
    ])
    Personal_Model = nn.Model(personal_sequential, loss = "mse")
    Personal_Model.fit(x_train,y_train,epochs = 1000, lr = 0.1)
    print(f"Personal model predictions : {Personal_Model.predict([[[1,1], [-1,1] , [-1,-1], [1,-1]]])}")
    