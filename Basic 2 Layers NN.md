# Neural Network - Basic 2 layers NN

Basic 2 layers neural network using sigmoid as activation function, and no bias:


```python
import numpy as np
import math
from typing import Callable, List
```


```python
class NeuralNetwork:
    """Neural Network class modeling a 2 layer NN without biases.

    Attributes:
        input                 (np.ndarray)               : Input training data.
        y                     (np.ndarray)               : Training data real output.
        n_features            (int)                      : Number of features (characteristics) in the training data.
        weights               (List[np.ndarray])         : Weights W with which are multiplied features before applying activation function.
        outputs               (List[np.ndarray])         : Output of each layer (Index 0 is layer 1, 1 is the output layer). Is updated at each training iteration.
        activation            (Callable[[float], float]) : Activation function of the neurons.
        activation_derivative (Callable[[float], float]) : Derivative of the activation function of the neurons.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, activation: Callable[[float], float], activation_derivative: Callable[[float], float]) -> NeuralNetwork:
        """
        Creates a neural network with the training data in input.

        Args:
            x (np.ndarray): Matrix NxM where N is the number of training examples and M the number of features.
            y (np.array)  : Array of training results of size N
        """
        self.input = x
        self.y = y
        self.n_features = x.shape[1]
        self.weights = [np.nan, np.nan] # Layer weights
        self.weights[0] = np.random.rand(self.n_features, 4)
        self.weights[1] = np.random.rand(4, 1)
        self.outputs = [np.zeros((4, 1)), np.zeros(y.shape)] # Layers outputs
        self.activation = activation
        self.activation_derivative = activation_derivative
        
    def feed_forward(self) -> np.ndarray:
        """
        Feed forward algorithm, dot product of layer input with its weights, and then applying activation function to it. Gives layers outputs.
        """
        self.outputs[0] = self.activation(np.dot(self.input, self.weights[0]))
        self.outputs[1] = self.activation(np.dot(self.outputs[0], self.weights[1]))
    
    def backward_propagation(self) -> None:
        """
        Backward propagation algorithm, using dervative of the cost function, with chain rule to include weight in it, to update weights.
        Calculus detail : https://www.youtube.com/watch?v=tIeHLnjs5U8
        """
        self.weights[1] += np.dot(self.outputs[0].T, (2 * (self.y - self.outputs[1])) * self.activation_derivative(self.outputs[1]))
        self.weights[0] += np.dot(self.input.T, (np.dot(2 * (self.y - self.outputs[1]) * self.activation_derivative(self.outputs[1]), self.weights[1].T) *  self.activation_derivative(self.outputs[0])))


def sigmoid(x: float, _lambda: float = 1) -> float:
    """
    Sigmoid function. https://en.wikipedia.org/wiki/Sigmoid_function
    
    Args:
        x       (float) : 
        _lambda (float) : (Prefixed with underscore because lambda is a reserved keyword of Python)
    """
    return 1 / (1 + np.exp(- _lambda * x))

def sigmoid_derivative(x: float, _lambda: float = 1) -> float:
    """
    Derivative of the sigmoid function.
    
    Args:
        x       (float) : 
        _lambda (float) : (Prefixed with underscore because lambda is a reserved keyword of Python)
    """
    return _lambda * x * (1 - x)
    

# Main
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # AND truth table
output = np.array([[0], [1], [1], [1]]) # AND truth table
n_iterations = 500
nn = NeuralNetwork(input, output, sigmoid, sigmoid_derivative)

# --> Training
for i in range(0, n_iterations):
    nn.feed_forward()
    nn.backward_propagation()

# --> Prediction
print(nn.outputs[1])
```

    [[0.05206804]
     [0.97087554]
     [0.97141422]
     [0.99192991]]
    

## Sources

[https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
