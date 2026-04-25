import pandas as pd
from sklearn.model_selection import train_test_split

def load_bank_data(data_path='data/Data/bank_additional_full.csv'):
    df = pd.read_csv(data_path, sep=';')
    return df

def preprocess_bank_data(df, drop_duration=True):
    """
    Handles encoding, dropping duration, and splitting features/target.
    """
    if drop_duration:
        df = df.drop('duration', axis=1)
    
    # Encode target
    df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop('y', axis=1)
    y = df_encoded['y']
    
    return X, y

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
