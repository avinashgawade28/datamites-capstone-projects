# Model Comparison Report: Insurance Cost Prediction

## 1. Introduction
This report compares the performance of multiple regression models in predicting individual medical insurance costs.

## 2. Models Evaluated
- **Linear Regression**: Baseline linear model.
- **Random Forest Regressor**: Ensemble of decision trees.
- **Gradient Boosting Regressor**: Boosting-based ensemble model.

## 3. Performance Metrics
| Model | R2 Score | MAE (Mean Absolute Error) | RMSE (Root Mean Square Error) |
|-------|----------|---------------------------|-------------------------------|
| Linear Regression | ~0.78 | ~4181 | ~5796 |
| Random Forest | ~0.86 | ~2550 | ~4581 |
| Gradient Boosting | ~0.88 | ~2448 | ~4345 |

## 4. Conclusion
The **Gradient Boosting Regressor** is the best model for this task, as it effectively captures the non-linear interactions between features like smoking status and BMI. It provides the most accurate cost predictions with the lowest error rates.
