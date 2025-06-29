# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Load and preprocess data
def load_data():
    # This is the exact dataset you provided
    data = """Store ID,Store_Area,Items_Available,Daily_Customer_Count,Store_Sales,Sales_per_SqFt,Customer_Density,High_Sales_Flag
1,1659,1961,530,66490,40.07836045810729,0.3194695599758891,0
2,1461,1752,210,39820,27.255304585900067,0.1437371663244353,0
3,1340,1609,720,54010,40.30597014925373,0.5373134328358209,0
... [ALL 896 ROWS OF YOUR DATA] ... 
896,1174,1429,1110,54340,46.28620102214651,0.9454855195911414,0"""
    
    from io import StringIO
    df = pd.read_csv(StringIO(data))
    return df

# Main training function
def train_and_save_model():
    print("Loading data...")
    df = load_data()
    
    # Data cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(columns=['Store ID', 'High_Sales_Flag'])
    
    # Feature engineering
    df['Sales_per_Customer'] = df['Store_Sales'] / df['Daily_Customer_Count']
    df['Area_Utilization'] = df['Items_Available'] / df['Store_Area']
    
    # Prepare data
    X = df.drop('Store_Sales', axis=1)
    y = df['Store_Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model R²: {r2:.4f}")
    print(f"MAE: ${mae:,.2f}")
    
    # Save artifacts
    joblib.dump(model, 'models/store_sales_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to models/ directory")

if __name__ == "__main__":
    train_and_save_model()