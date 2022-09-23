from random import randint
from typing import List
import numpy as np
from data_loader import DataLoader

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

if __name__ == "__main__":
    nn = MLP([784, 16, 16, 10])
    data = DataLoader()
    img = data.training_data[randint(0, 30000)][0]
    result = nn.predict(img)
    print(result)

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    plt.imshow(img.reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
