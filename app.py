import streamlit as st
import pandas as pd
from joblib import load

# Load the saved XGBoost model

model = load('models/xgboost_classifer.clf')


# Function to preprocess user inputs
def preprocess_input(age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, test_troponin):
    # Convert gender to numerical value
    gender = 0 if gender == 'Female' else 1
    # Other preprocessing steps if needed
    return [[age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, test_troponin]]

# Streamlit web app
st.title('Heart Attack Prediction')
st.write('Enter the required information below:')

# Create input fields for user input
age = st.number_input('Age', min_value=0, max_value=150, value=50)
gender = st.selectbox('Gender', ['Female', 'Male'])
heart_rate = st.number_input('Heart Rate', min_value=0, max_value=300, value=70)
systolic_bp = st.number_input('Systolic BP', min_value=0, max_value=300, value=120)
diastolic_bp = st.number_input('Diastolic BP', min_value=0, max_value=200, value=80)
blood_sugar = st.number_input('Blood Sugar', min_value=0, max_value=500, value=100)
ck_mb = st.number_input('CK-MB', min_value=0, max_value=100, value=10)
test_troponin = st.slider('Test Troponin', min_value=0.0, max_value=5.0, value=0.0, format="%.4f")


# Create a button to predict
if st.button('Predict'):
    # Preprocess user inputs
    input_data = preprocess_input(age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, test_troponin)

    # Make a prediction using the trained model
    prediction = model.predict(input_data)

    # Display the prediction result
    if prediction[0] == 1:
        st.write('Prediction: Positive - Presence of Heart Attack')
    else:
        st.write('Prediction: Negative - Absence of Heart Attack')

