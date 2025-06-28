# === 1. Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# === 2. Data Loading & Cleaning ===
df = pd.read_csv('store_sales.csv')

# Handle missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert to numeric types
numeric_cols = ['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Remove irrelevant columns
df = df.drop(columns=['Store ID', 'High_Sales_Flag'])

# Handle outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# === 3. EDA ===
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Target distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Store_Sales'], kde=True)
plt.title('Store Sales Distribution')
plt.show()

# === 4. Feature Engineering ===
# Create new features
df['Sales_per_Customer'] = df['Store_Sales'] / df['Daily_Customer_Count']
df['Area_Utilization'] = df['Items_Available'] / df['Store_Area']

# Prepare data for modeling
X = df.drop('Store_Sales', axis=1)
y = df['Store_Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Model Building ===
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    results[name] = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

# Compare performance
results_df = pd.DataFrame(results).T
print(results_df.sort_values('R2', ascending=False))

# Hyperparameter tuning (example for Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# === 6. Model Evaluation ===
y_pred_best = best_model.predict(X_test_scaled)
print(f"Best Model R2: {r2_score(y_test, y_pred_best):.4f}")
print(f"Best Model MAE: {mean_absolute_error(y_test, y_pred_best):.2f}")

# Feature importance
feature_importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.show()

# === 7. Save Model ===
joblib.dump(best_model, 'store_sales_model.pkl')
joblib.dump(scaler, 'scaler.pkl')