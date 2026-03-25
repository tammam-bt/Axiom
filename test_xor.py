import NN
# import tensorflow as tf
import time
import numpy as np
np.random.seed(42)
if __name__ == "__main__":
    # Training data (4 samples, 2 features each)
    x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    print(x_train.shape)
    # Targets (4 samples, 1 output each)
    y_train = np.array([[0], [1], [1], [0]])
    # start_time = time.perf_counter()
    # tf_model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units = 16, activation = "tanh"),
    #     tf.keras.layers.Dense(units = 15, activation = "relu"),
    #     tf.keras.layers.Dense(units = 1, activation="sigmoid")
    # ])
    # tf_model.compile(loss = tf.keras.losses.BinaryCrossentropy)
    # tf_model.fit(x_train,y_train, epochs = 1000)
    # end_time = time.perf_counter()
    # tf_time = end_time - start_time
    start_time = time.perf_counter()
    personal_sequential = NN.Sequential([
        NN.Dense(2,3),
        NN.Sigmoid(),
        NN.Dense(3,1),
        NN.Sigmoid()
    ])
    Personal_Model = NN.Model(personal_sequential, loss = "bce")
    Personal_Model.fit(x_train,y_train,epochs = 5000, lr = 1)
    end_time = time.perf_counter()
    personal_time = end_time - start_time
    print(f"Personal model predictions : {Personal_Model.predict([[0,0],[0,1],[1,0],[1,1]])}")
    # print(f"Tensorflow model predictions : {tf_model.predict(np.array([[0,0],[0,1],[1,0],[1,1]]))}")
    print(f"Personal Time: {personal_time}s")
    # print(f"Tensor Flow Model Time: {tf_time}s")
    