import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the advertising dataset
    """
    # Load the data
    df = pd.read_csv(file_path, index_col=0)
    
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print(df.columns)
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")
    
    # Check for outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print(f"\nOutliers detected: {outliers}")
    
    return df

def prepare_features_target(df):
    """
    Prepare features and target variable
    """
    # Feature selection
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    return X, y

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Test the functions
    df = load_and_preprocess_data('E:\Code alpha Internship\Sales Prediction using python\Advertising.csv')
    X, y = prepare_features_target(df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")