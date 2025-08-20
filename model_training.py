import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_models(X, y, test_size=0.2, random_state=42):
    """
    Train multiple regression models and evaluate them
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'R² Score': r2,
            'Model': model
        }
        
        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print("-" * 40)
    
    # Select best model
    best_model_name = min(results, key=lambda x: results[x]['RMSE'])
    best_model = results[best_model_name]['Model']
    print(f"\nBest Model: {best_model_name}")
    
    # Save the best model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Best model and scaler saved as 'best_model.pkl' and 'scaler.pkl'")
    
    return results, best_model, scaler, X_test_scaled, y_test

def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance for the model
    """
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return feature_importance
        
    elif hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nCoefficient Analysis:")
        print(coef_df)
        
        return coef_df

if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv('E:\Code alpha Internship\Sales Prediction using python\Advertising.csv', index_col=0)
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Train models
    results, best_model, scaler, X_test_scaled, y_test = train_models(X, y)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(best_model, X.columns)