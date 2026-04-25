# Report on Challenges Faced: Heart Disease Prediction

## 1. Small Dataset Size
- **Challenge**: With only 180 patients, the model is prone to high variance. Small changes in the training set can significantly affect accuracy.
- **Technique**: Used stratified k-fold cross-validation (conceptually in testing) and held-out validation sets to ensure robust results.

## 2. Categorical Variables
- **Challenge**: The `thal` feature is categorical (normal, fixed defect, reversible defect).
- **Technique**: Applied One-Hot Encoding to convert categories into a format suitable for machine learning algorithms.

## 3. Clinical Suggestions
- **Challenge**: Translating model results into actionable hospital advice.
- **Technique**: Analyzed feature importance to highlight key indicators like `thal` and `max_heart_rate_achieved` for hospital staff.
