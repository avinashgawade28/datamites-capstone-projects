import os
import argparse
import joblib
from data_loader import load_heart_data, preprocess_heart_data, get_train_test_split
from models import build_logistic_regression, build_random_forest, build_svm
from sklearn.metrics import accuracy_score, classification_report, f1_score

def main():
    parser = argparse.ArgumentParser(description="Train models for Heart Disease Prediction")
    parser.add_argument("--model", type=str, default="rf", choices=["lr", "rf", "svm"], help="Model to train")
    args = parser.parse_args()

    # Get the project root directory (one level up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "Data")

    print(f"Loading data from {data_dir}...")
    df = load_heart_data(data_dir)
    X, y = preprocess_heart_data(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    if args.model == "lr":
        print("Training Logistic Regression...")
        model = build_logistic_regression()
    elif args.model == "svm":
        print("Training SVM...")
        model = build_svm()
    else: # rf
        print("Training Random Forest...")
        model = build_random_forest()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"\nModel: {args.model}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Save model
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"heart_disease_{args.model}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
