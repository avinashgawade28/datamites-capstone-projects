import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    """
    Loads MNIST dataset and returns (x_train, y_train), (x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test, flatten=False):
    """
    Normalizes images to [0, 1] and optionally flattens them.
    """
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
    return x_train, x_test
