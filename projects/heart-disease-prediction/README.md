# Heart Disease Prediction (PRCP-1016)

## Project Overview
This project aims to predict the presence of heart disease in patients based on 13 clinical features. Early detection of cardiovascular diseases (CVDs) is crucial for managing high-risk patients and preventing premature deaths.

## Dataset
The dataset contains 180 samples with 13 features and 1 target variable (`heart_disease_present`).
- `values.csv`: Clinical features for each patient.
- `labels.csv`: Ground truth labels (0: No disease, 1: Disease present).

## Key Tasks
1. **Data Analysis**: Explore relationships between clinical features and heart disease.
2. **Predictive Modeling**: Train and evaluate machine learning models (Logistic Regression, Random Forest, SVM).
3. **Clinical Suggestions**: Provide actionable insights for healthcare providers.

## Project Structure
- `data/`: Contains the raw CSV files.
- `notebooks/`: Main analysis and modeling notebook.
- `reports/`: Model comparison and challenge reports.
- `src/`: Modular Python source code for data loading, modeling, and training.
- `requirements.txt`: List of dependencies.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook notebooks/heart_disease_prediction.ipynb`
3. Or use the modular scripts:
   - `python src/train.py --model rf` (Random Forest)
   - `python src/train.py --model lr` (Logistic Regression)
