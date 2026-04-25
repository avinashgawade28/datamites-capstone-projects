# Handwritten Digits Image Processing (PRCP-1002)

## Project Overview
The goal of this project is to correctly identify digits from a dataset of tens of thousands of handwritten images (MNIST dataset). This project involves exploring various machine learning algorithms, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Simple Neural Networks, to compare their performance.

## Dataset
The project uses the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9) of size 28x28 pixels.
- Training set: 60,000 images
- Test set: 10,000 images

## Skills Practiced
- Computer vision fundamentals
- Simple neural networks
- Classification methods (SVM, KNN)
- Model evaluation and comparison

## Project Structure
- `data/`: Contains dataset information or scripts to load data.
- `notebooks/`: Contains the main Jupyter notebook for the project.
- `reports/`: Contains model comparison and challenge reports.
- `src/`: Source code for the project (if any).
- `requirements.txt`: List of dependencies.

## How to Run

### Using Jupyter Notebook
1. Install dependencies: `pip install -r requirements.txt`
2. Open the Jupyter notebook: `jupyter notebook notebooks/handwritten_digits_recognition.ipynb`
3. Run all cells in the notebook.

### Using Modular Scripts (src/)
You can also run the training pipeline using the modular scripts in the `src/` directory:
1. Navigate to the project directory.
2. Train a specific model:
   - Neural Network: `python src/train.py --model nn`
   - SVM: `python src/train.py --model svm --subset 5000`
   - KNN: `python src/train.py --model knn --subset 5000`
3. Trained models will be saved in the `models/` directory.
