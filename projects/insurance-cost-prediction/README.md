# Insurance Cost Prediction (PRCP-1021)

## Project Overview
This project involves building a machine learning model to predict individual medical costs billed by health insurance. The prediction is based on features such as age, gender, BMI, number of children, smoking status, and region.

## Dataset
- `insurance.csv`: Contains 1338 observations with 7 features.
- Target variable: `charges`.

## Features
- `age`: Age of primary beneficiary.
- `sex`: Gender (male/female).
- `bmi`: Body mass index.
- `children`: Number of dependents.
- `smoker`: Smoking status (yes/no).
- `region`: Residential area in the US.
- `charges`: Individual medical costs.

## Project Structure
- `data/`: Contains the raw dataset.
- `notebooks/`: Main Jupyter notebook for EDA and modeling.
- `reports/`: Model comparison and challenge reports.
- `src/`: Modular Python scripts for data processing and training.
- `requirements.txt`: Dependencies.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook notebooks/insurance_cost_prediction.ipynb`
3. Run modular scripts:
   - `python src/train.py --model linear` (Linear Regression)
   - `python src/train.py --model rf` (Random Forest Regressor)
