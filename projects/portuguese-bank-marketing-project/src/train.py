import os
import argparse
import joblib
import pandas as pd
import numpy as np
from data_loader import load_bank_data, preprocess_bank_data, get_train_test_split
from models import build_logistic_regression, build_random_forest, build_gradient_boosting
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

def main():
    parser = argparse.ArgumentParser(description="Train models for Portuguese Bank Marketing")
    parser.add_argument("--model", type=str, default="rf", choices=["lr", "rf", "gb"], help="Model to train")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "Data", "bank_additional_full.csv")

    print(f"Loading data from {data_path}...")
    df = load_bank_data(data_path)
    X, y = preprocess_bank_data(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    if args.model == "lr":
        print("Training Logistic Regression...")
        model = build_logistic_regression()
    elif args.model == "gb":
        print("Training Gradient Boosting...")
        model = build_gradient_boosting()
    else: # rf
        print("Training Random Forest...")
        model = build_random_forest()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\nModel: {args.model}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, preds):.4f}")
    if probs is not None:
        print(f"ROC AUC: {roc_auc_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Save model
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"bank_marketing_{args.model}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
