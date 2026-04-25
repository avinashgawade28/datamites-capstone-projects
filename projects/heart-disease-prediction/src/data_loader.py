import pandas as pd
from sklearn.model_selection import train_test_split

def load_heart_data(data_dir='data/Data'):
    """
    Loads values and labels, merges them on patient_id.
    """
    values = pd.read_csv(f"{data_dir}/values.csv")
    labels = pd.read_csv(f"{data_dir}/labels.csv")
    df = pd.merge(values, labels, on='patient_id')
    return df

def preprocess_heart_data(df):
    """
    Handles categorical encoding and feature scaling.
    """
    # Drop patient_id
    df = df.drop('patient_id', axis=1)
    
    # One-hot encode 'thal'
    df = pd.get_dummies(df, columns=['thal'], drop_first=True)
    
    X = df.drop('heart_disease_present', axis=1)
    y = df['heart_disease_present']
    
    return X, y

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
