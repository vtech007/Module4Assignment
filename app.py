import streamlit as st
import joblib
import pandas as pd
import numpy as np 

# Load model and scaler
model = joblib.load('store_sales_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Store Sales Prediction')

# Input widgets
store_area = st.number_input('Store Area (sq ft)', min_value=0)
items_available = st.number_input('Items Available', min_value=0)
daily_customers = st.number_input('Daily Customers', min_value=0)
sales_per_sqft = st.number_input('Sales per SqFt', min_value=0.0)
customer_density = st.number_input('Customer Density', min_value=0.0)

if st.button('Predict Sales'):
    # Create feature array
    features = pd.DataFrame([[store_area, items_available, daily_customers, 
                            sales_per_sqft, customer_density]],
                            columns=['Store_Area', 'Items_Available', 
                                    'Daily_Customer_Count', 'Sales_per_SqFt',
                                    'Customer_Density'])
    
    # Generate derived features
    features['Sales_per_Customer'] = features['Items_Available'] / features['Store_Area']
    features['Area_Utilization'] = features['Items_Available'] / features['Store_Area']
    
    # Scale features and predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    st.success(f'Predicted Store Sales: ${prediction:,.2f}')