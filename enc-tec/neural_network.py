import random
from typing import List
import numpy as np
from data_loader import DataLoader
from secret import backpropagation

class MLP:
    """TODO."""

    def __init__(self, layer_sizes: List[int]) -> None:
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    
    def predict(self, a) -> int:
        return np.argmax([x[0] for x in self._feedforward(a)])
    
    def _feedforward(self, a) -> List[List[float]]:
        for w, b in zip(self.weights, self.biases):
            a = self._calculate_layer_output(a, w, b)
        return a
    
    def _calculate_layer_output(self, input, weights, biases):
        return self._sigmoid(np.dot(weights, input) + biases)
    
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def encode_class(class_value: int) -> np.array:
        encoded_class = np.zeros((10, 1))
        encoded_class[class_value] = 1.0
        return encoded_class


    # Training methods

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f'Epoch {i}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {i} completed')
    
    def update_mini_batch(self, mini_batch, learning_rate):
        """Updates network weights and biases by applying gradient descent."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = backpropagation(self.weights,
                self.biases, self.number_of_layers, x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self._feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

if __name__ == "__main__":
    nn = MLP([784, 16, 16, 10])
    data = DataLoader()
    a = []
    b = []
    for t in data.training_data:
        a.append(t[0])
        b.append(t[1])
    training_x = a
    training_y = [MLP.encode_class(x) for x in b]
    training_data = list(zip(training_x, training_y))
    training_data = [(t[0], MLP.encode_class(t[1])) for t in data.training_data]
    nn.SGD(training_data, 30, 10, 3.0, test_data=data.test_data)
