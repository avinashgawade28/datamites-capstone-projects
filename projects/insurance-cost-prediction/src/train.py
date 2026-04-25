import os
import argparse
import joblib
import pandas as pd
import numpy as np
from data_loader import load_insurance_data, preprocess_insurance_data, get_train_test_split
from models import build_linear_regression, build_random_forest, build_gradient_boosting
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser(description="Train models for Insurance Cost Prediction")
    parser.add_argument("--model", type=str, default="rf", choices=["linear", "rf", "gb"], help="Model to train")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "Data", "insurance.csv")

    print(f"Loading data from {data_path}...")
    df = load_insurance_data(data_path)
    X, y = preprocess_insurance_data(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    if args.model == "linear":
        print("Training Linear Regression...")
        model = build_linear_regression()
    elif args.model == "gb":
        print("Training Gradient Boosting...")
        model = build_gradient_boosting()
    else: # rf
        print("Training Random Forest...")
        model = build_random_forest()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"\nModel: {args.model}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    print(f"R2 Score: {r2_score(y_test, preds):.4f}")
    
    # Save model
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"insurance_cost_{args.model}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
