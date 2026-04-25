from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_knn(n_neighbors=3):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def build_svm(kernel='rbf', C=1.0):
    return SVC(kernel=kernel, C=C)

def build_neural_network(input_shape=(28, 28), num_classes=10):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
