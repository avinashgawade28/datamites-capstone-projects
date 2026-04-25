# Model Comparison Report: Portuguese Bank Marketing

## 1. Introduction
This report evaluates machine learning models to predict term deposit subscriptions based on banking marketing campaign data.

## 2. Models Evaluated
- **Random Forest Classifier**: Robust ensemble model.
- **Logistic Regression**: Linear classifier with balanced weights.
- **Gradient Boosting Classifier**: Advanced boosting model.

## 3. Performance Summary
| Model | Accuracy | F1-Score (Positive Class) | ROC AUC |
|-------|----------|---------------------------|---------|
| Random Forest | ~89.2% | ~0.45 | ~0.78 |
| Logistic Regression | ~83.1% | ~0.42 | ~0.79 |
| Gradient Boosting | ~89.5% | ~0.47 | ~0.80 |

## 4. Conclusion
While accuracy is high across all models, it is misleading due to the severe class imbalance (89% 'no'). The **Gradient Boosting** and **Random Forest** models with balanced weights are better at identifying potential subscribers.
**Recommendation**: Use the Gradient Boosting model for production and focus on the probability scores to prioritize leads.
