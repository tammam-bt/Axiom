<img src="https://cdn-icons-png.flaticon.com/512/7747/7747363.png" alt="Logo of the project" height="100">

# Neural Network Library from Scratch
> A minimal NumPy-based neural network framework to understand ML internals

This project is a lightweight neural network library implemented entirely from scratch using Python and NumPy. Its primary goal is educational: to expose the inner mechanics of neural networks such as forward propagation, backpropagation, gradient descent, and layer abstraction.

Instead of relying on high-level frameworks like TensorFlow or PyTorch, this library focuses on clarity and simplicity, making it ideal for learning how neural networks actually work under the hood.

---

## Installing / Getting started

Minimal setup to run the project:

```bash
pip install numpy matplotlib
```

Clone the repository and run the example:

```bash
git clone https://github.com/tammam-bt/NN-library-from-scratch.git
cd NN-library-from-scratch
python test_xor.py
```

This will train a small neural network on the XOR problem and output predictions.

---

### Initial Configuration

No special configuration is required. The only dependencies are:

- `numpy`
- `matplotlib` (optional, for visualization)

---

## Developing

To start modifying or extending the library:

```bash
git clone https://github.com/tammam-bt/NN-library-from-scratch.git
cd NN-library-from-scratch
```

Project structure is simple:
- `NN.py` → Core library implementation
- `test_xor.py` → Example usage (XOR problem)

Modify or extend classes such as `Dense`, `Activation`, or `Model` to experiment with new architectures or learning techniques.

---

### Building

No build step is required since this is a pure Python project.

Running scripts directly executes the code:

```bash
python test_xor.py
```

---

### Deploying / Publishing

This project is not packaged for distribution yet. Future improvements may include publishing to PyPI.

---

## Features

- Fully connected (`Dense`) layer implementation
- Custom activation functions:
  - Sigmoid
  - Tanh
- Forward propagation
- Backpropagation with gradient descent
- Mean Squared Error (MSE) loss
- Sequential model composition
- Simple training loop (`fit`)
- XOR problem demonstration

---

## Configuration

### Model.fit Parameters

#### `epochs`
Type: `int`  
Default: None  

Number of training iterations over the dataset.

Example:
```python
Model.fit(x_train, y_train, epochs=10000, lr=0.1)
```

---

#### `lr` (learning rate)
Type: `float`  
Default: None  

Controls how much weights are updated during training.

---

## Example Usage

```python
import NN
import numpy as np

np.random.seed(42)

x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0], [1], [1], [0]])

model = NN.Model(
    NN.Sequential(
        NN.Dense(2,3),
        NN.Tanh(),
        NN.Dense(3,1),
        NN.Sigmoid()
    )
)

model.fit(x_train, y_train, epochs=10000, lr=0.1)

print(model.predict(x_train))
```

---

## Future Features

Planned improvements include:

- Additional activation functions (ReLU, Leaky ReLU)
- More loss functions (Cross-Entropy)
- Optimizers (Adam, RMSprop)
- Mini-batch training
- Model saving/loading
- Visualization tools for training

---

## Contributing

If you'd like to contribute, fork the repository and create a feature branch. Pull requests are welcome.

For major changes, consider opening an issue first to discuss the approach.

---

## Links

- Repository: https://github.com/tammam-bt/NN-library-from-scratch
- Issue Tracker: https://github.com/tammam-bt/NN-library-from-scratch/issues

---

## Licensing

The code in this project is licensed under the MIT License.
