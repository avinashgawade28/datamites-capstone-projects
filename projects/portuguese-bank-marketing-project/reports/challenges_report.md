# Report on Challenges Faced: Portuguese Bank Marketing

## 1. Severe Class Imbalance
- **Challenge**: Only about 11% of the customers subscribed to the term deposit. Standard models tend to predict the majority class ('no') for all instances.
- **Technique**: Used `class_weight='balanced'` in Random Forest and Logistic Regression. Explored ROC AUC as a more reliable metric than Accuracy.

## 2. Realistic Modeling (Duration Attribute)
- **Challenge**: The `duration` attribute is highly predictive but not available before a call. Including it makes the model unrealistic.
- **Technique**: Discarded the `duration` column for the final predictive model as per the problem statement's suggestion.

## 3. High Cardinality of Categorical Features
- **Challenge**: Features like `job` and `education` have many levels.
- **Technique**: Used One-Hot Encoding and analyzed which specific jobs (e.g., retired, student) were most impactful.
