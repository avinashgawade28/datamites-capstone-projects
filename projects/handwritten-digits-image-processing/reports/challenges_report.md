# Report on Challenges Faced: Handwritten Digits Recognition

## 1. Data Dimensionality
- **Challenge**: Each image is 28x28 pixels, resulting in 784 features. Traditional models like KNN can suffer from the "curse of dimensionality".
- **Technique**: Flattened the images and explored normalization to ensure features were on the same scale.

## 2. Computational Complexity
- **Challenge**: SVM training on 60,000 samples is computationally expensive (O(n²)).
- **Technique**: Used a subset of the data for initial model selection and hyperparameter tuning to save time.

## 3. Feature Scaling
- **Challenge**: Raw pixel values (0-255) can lead to large gradients and slow convergence in neural networks.
- **Technique**: Normalized pixel values to the [0, 1] range by dividing by 255.0.

## 4. Hyperparameter Tuning
- **Challenge**: Finding the right `k` for KNN or the right `C` and `gamma` for SVM.
- **Technique**: Used cross-validation (implied in the notebook experimentation) to find optimal parameters.
