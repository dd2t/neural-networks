import gzip
import pickle
from typing import Tuple
import numpy as np

class DataLoader:
    """TODO."""

    @property
    def training_data(self) -> zip:
        if not self.__is_dataset_loaded():
            self._load_dataset()
        return self._training_data
    
    @property
    def validation_data(self) -> zip:
        if not self.__is_dataset_loaded():
            self._load_dataset()
        return self._validation_data
    
    @property
    def test_data(self) -> zip:
        if not self.__is_dataset_loaded():
            self._load_dataset()
        return self._test_data

    def __init__(self) -> None:
        self._training_data: zip = None
        self._validation_data: zip = None
        self._test_data: zip = None

        if not self.__is_dataset_loaded():
            self._load_dataset()
    
    def _load_dataset(self) -> None:
        print('Loading MNIST dataset...')
        with gzip.open('./data/mnist.pkl.gz', 'rb') as pickled_file:
            training_set, validation_set, test_set = pickle.load(pickled_file,
                                                                 encoding='latin1')
            self._prepare_datasets(training_set, validation_set, test_set)

    def _prepare_datasets(self, training_set: Tuple, validation_set: Tuple,
                         test_set: Tuple) -> None:
        training_inputs = [np.reshape(x, (784, 1)) for x in training_set[0]]
        validation_inputs = [np.reshape(x, (784, 1)) for x in validation_set[0]]
        test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]

        self._training_data = list(zip(training_inputs, training_set[1]))
        self._validation_data = list(zip(validation_inputs, validation_set[1]))
        self._test_data = list(zip(test_inputs, test_set[1]))
    
    def __is_dataset_loaded(self) -> bool:
        datasets = (self._training_data, self._validation_data, self._test_data)
        if any(map(lambda x: x is None, datasets)):
            return False
        return True

if __name__ == "__main__":
    DataLoader()