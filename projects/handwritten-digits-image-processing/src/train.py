import os
import argparse
from data_loader import load_mnist_data, preprocess_data
from models import build_knn, build_svm, build_neural_network
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    parser = argparse.ArgumentParser(description="Train models for Handwritten Digits Recognition")
    parser.add_argument("--model", type=str, default="nn", choices=["knn", "svm", "nn"], help="Model to train")
    parser.add_argument("--subset", type=int, default=10000, help="Number of training samples to use for ML models")
    args = parser.parse_args()

    print(f"Loading data...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    if args.model in ["knn", "svm"]:
        print(f"Preprocessing for {args.model} (flattening and normalization)...")
        x_train_proc, x_test_proc = preprocess_data(x_train, x_test, flatten=True)
        # Use subset for ML models as they are slower
        x_train_proc = x_train_proc[:args.subset]
        y_train = y_train[:args.subset]
        x_test_proc = x_test_proc[:2000] # Test on a subset for speed
        y_test = y_test[:2000]
    else:
        print(f"Preprocessing for Neural Network (normalization)...")
        x_train_proc, x_test_proc = preprocess_data(x_train, x_test, flatten=False)

    if args.model == "knn":
        print("Training KNN...")
        model = build_knn()
        model.fit(x_train_proc, y_train)
        preds = model.predict(x_test_proc)
        acc = accuracy_score(y_test, preds)
    
    elif args.model == "svm":
        print("Training SVM...")
        model = build_svm()
        model.fit(x_train_proc, y_train)
        preds = model.predict(x_test_proc)
        acc = accuracy_score(y_test, preds)
    
    else: # nn
        print("Training Neural Network...")
        model = build_neural_network()
        model.fit(x_train_proc, y_train, epochs=5, batch_size=32, validation_split=0.1)
        loss, acc = model.evaluate(x_test_proc, y_test)

    print(f"\nModel: {args.model}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    if args.model == "nn":
        model.save(f"models/handwritten_digits_{args.model}.h5")
    else:
        joblib.dump(model, f"models/handwritten_digits_{args.model}.pkl")
    print(f"Model saved to models/handwritten_digits_{args.model}")

if __name__ == "__main__":
    main()
