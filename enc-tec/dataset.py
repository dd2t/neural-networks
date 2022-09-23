import gzip
import pickle
import numpy as np
from random import randint
import matplotlib.cm as cm
import matplotlib.pyplot as plt

with gzip.open('../data/mnist.pkl.gz', 'rb') as pickled_file:
    train_set, valid_set, test_set = pickle.load(pickled_file, encoding='latin1')

training_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
training_results = train_set[1]

# print(training_results[0])

plt.imshow(training_inputs[randint(0, 30000)].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()
