# Portuguese Bank Marketing Project (PRCP-1000)

## Project Overview
This project aims to predict whether a customer will subscribe to a term deposit based on a direct marketing campaign (phone calls) conducted by a Portuguese banking institution.

## Dataset
- `bank_additional_full.csv`: Contains 41,188 observations and 20 input variables.
- Target variable: `y` (binary: 'yes', 'no').

## Key Variables
- **Client Data**: age, job, marital status, education, credit default, housing loan, personal loan.
- **Campaign Data**: contact type, month, day of week, duration (to be dropped for realistic modeling).
- **Social & Economic Context**: employment variation rate, consumer price index, euribor 3 month rate, etc.

## Project Structure
- `data/`: Contains the raw dataset.
- `notebooks/`: Main Jupyter notebook for EDA and predictive modeling.
- `reports/`: Model comparison and challenge reports.
- `src/`: Modular Python scripts for data processing and training.
- `requirements.txt`: Dependencies.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook notebooks/bank_marketing_analysis.ipynb`
3. Run modular scripts:
   - `python src/train.py --model rf` (Random Forest)
   - `python src/train.py --model lr` (Logistic Regression)
