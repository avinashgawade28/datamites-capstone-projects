import pandas as pd
from sklearn.model_selection import train_test_split

def load_insurance_data(data_path='data/Data/insurance.csv'):
    df = pd.read_csv(data_path)
    return df

def preprocess_insurance_data(df):
    """
    Handles encoding of categorical variables.
    """
    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    
    return X, y

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
