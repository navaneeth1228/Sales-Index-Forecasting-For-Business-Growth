import warnings
warnings.filterwarnings('ignore')

# +
import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

# Load the saved XGBoost model
model_path = "xg_reg_model.pkl"
xg_reg = joblib.load(model_path)

# Title of the app
st.title("Sales Index Prediction App")

# Define the feature names (must match the model's feature names)
feature_names = ['Year', 'Month', 'Transactional Index_normalized', 'Sales YOY %_normalized', 'Transaction YOY %_normalized'] + \
                [f"State_{state}" for state in ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 
                                                'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 
                                                'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 
                                                'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 
                                                'VA', 'VT', 'WA', 'WI', 'WV', 'WY']] + \
                [f"Sector Name_{sector}" for sector in ['Accommodation and Food Services', 
                                                        'Administrative and Environmental Services', 
                                                        'Arts, Entertainment, and Recreation', 'Construction', 
                                                        'Educational Services', 'Finance and Insurance', 
                                                        'Health Care and Social Assistance', 
                                                        'Information', 'Manufacturing', 
                                                        'Other Services (except Public Administration)', 
                                                        'Professional, Scientific, and Technical Services', 
                                                        'Real Estate and Rental and Leasing', 'Retail', 
                                                        'Transportation and Warehousing', 'Utilities', 
                                                        'Wholesale Trade']]

# Create a form for user inputs
with st.form("input_form"):
    st.header("Enter Feature Values")
    
    # Input for Year and Month
    input_data = {}
    input_data['Year'] = st.number_input('Enter Year', value=2024, step=1)
    input_data['Month'] = st.number_input('Enter Month (1-12)', value=1, step=1)
    
    # Input for transactional index and YOY percentages
    transactional_index = st.number_input('Enter Transactional Index', value=10.0, step=0.1)
    sales_yoy = st.number_input('Enter Sales Year over Year (%)', value=10.0, step=0.1)
    transaction_yoy = st.number_input('Enter Transaction Year over Year (%)', value=10.0, step=0.1)
    
    # Normalize the inputs
    transactional_min = 2.13
    transactional_max = 286.635
    input_data['Transactional Index_normalized'] = (transactional_index - transactional_min) / (transactional_max - transactional_min)
    
    sales_yoy_min = -96.96
    sales_yoy_max = 153.91
    input_data['Sales YOY %_normalized'] = (sales_yoy - sales_yoy_min) / (sales_yoy_max - sales_yoy_min)
    
    transaction_yoy_min = -98.015
    transaction_yoy_max = 146.73
    input_data['Transaction YOY %_normalized'] = (transaction_yoy - transaction_yoy_min) / (transaction_yoy_max - transaction_yoy_min)
    
    # State selection
    states = [state for state in ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 
                                  'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 
                                  'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 
                                  'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 
                                  'VA', 'VT', 'WA', 'WI', 'WV', 'WY']]
    state_option = st.selectbox("Select State", states)
    for state in states:
        input_data[f"State_{state}"] = 1 if state == state_option else 0

    # Sector selection
    sectors = ['Accommodation and Food Services', 
               'Administrative and Environmental Services', 
               'Arts, Entertainment, and Recreation', 'Construction', 
               'Educational Services', 'Finance and Insurance', 
               'Health Care and Social Assistance', 'Information', 
               'Manufacturing', 'Other Services (except Public Administration)', 
               'Professional, Scientific, and Technical Services', 
               'Real Estate and Rental and Leasing', 'Retail', 
               'Transportation and Warehousing', 'Utilities', 
               'Wholesale Trade']
    sector_option = st.selectbox("Select Sector", sectors)
    for sector in sectors:
        input_data[f"Sector Name_{sector}"] = 1 if sector == sector_option else 0
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# Handle prediction
if submitted:
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure the input_df columns match the model's feature names
    input_df = input_df[feature_names]
    
    # Predict sales index
    prediction_normalized = xg_reg.predict(input_df)[0]
    
    # Denormalize the prediction
    sales_index_min = 3.38
    sales_index_max = 285.135
    prediction = prediction_normalized * (sales_index_max - sales_index_min) + sales_index_min
    
    # Display the result
    st.subheader("Predicted Sales Index")
    st.write(prediction)


