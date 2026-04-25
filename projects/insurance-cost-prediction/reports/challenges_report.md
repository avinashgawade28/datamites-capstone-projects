# Report on Challenges Faced: Insurance Cost Prediction

## 1. Feature Interaction
- **Challenge**: The impact of BMI on insurance charges is significantly amplified for smokers. A simple linear model struggles to capture this interaction without manual feature engineering.
- **Technique**: Used tree-based ensemble models (Random Forest and Gradient Boosting) which inherently handle feature interactions.

## 2. Categorical Encoding
- **Challenge**: Features like `region` have multiple categories.
- **Technique**: Used One-Hot Encoding with `drop_first=True` to avoid the dummy variable trap.

## 3. Skewed Target Variable
- **Challenge**: The `charges` variable is right-skewed.
- **Technique**: While not strictly necessary for tree models, future work could involve log-transforming the target variable to stabilize variance for linear models.
