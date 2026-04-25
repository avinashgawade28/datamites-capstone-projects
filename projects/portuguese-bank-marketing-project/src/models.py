from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_logistic_regression():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

def build_random_forest(n_estimators=100):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')

def build_gradient_boosting():
    return GradientBoostingClassifier(random_state=42)
