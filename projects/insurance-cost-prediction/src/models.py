from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_linear_regression():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])

def build_random_forest(n_estimators=100):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=42)

def build_gradient_boosting():
    return GradientBoostingRegressor(random_state=42)
