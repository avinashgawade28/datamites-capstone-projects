# Model Comparison Report: Handwritten Digits Recognition

## 1. Introduction
This report compares the performance of three different machine learning models on the MNIST handwritten digits dataset.

## 2. Models Evaluated
- **K-Nearest Neighbors (KNN)**: A simple instance-based learning algorithm.
- **Support Vector Machine (SVM)**: A powerful classifier that uses the kernel trick to handle high-dimensional data.
- **Simple Neural Network (MLP)**: A multi-layer perceptron with two hidden layers.

## 3. Performance Comparison
| Model | Training Accuracy | Test Accuracy | Training Time |
|-------|-------------------|---------------|---------------|
| KNN   | High              | ~95%          | Low (Lazy)    |
| SVM   | Very High         | ~96%          | High          |
| MLP   | Very High         | ~98%          | Moderate      |

## 4. Conclusion and Recommendation
The **Neural Network (MLP)** outperformed the traditional machine learning models. It provides the best balance between accuracy and inference speed once trained.
**Recommendation**: Use the Neural Network model for production. For further improvement, a Convolutional Neural Network (CNN) is recommended as it is specifically designed for image data.
