import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Medical Cost Predictor", layout="centered")

# Custom CSS for green bar and proper spacing
st.markdown("""
<style>
    .main > div {
        margin-top: 2rem;
    }
    .prediction-bar {
        background-color: #2ecc71;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .prediction-label {
        font-size: 1.2rem;
        font-weight: normal;
        color: white;
        margin-bottom: 0.2rem;
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: bold;
        color: white;
        line-height: 1;
    }
    div.row-widget.stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }
    .stSlider > div {
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('ols_model.pkl')
    return scaler, model

try:
    scaler, model = load_models()
except Exception as e:
    st.error("Error loading model files. Make sure 'scaler.pkl' and 'ols_model.pkl' are in the same folder.")
    st.stop()

# Function to make prediction
def predict(age, bmi, children, sex, smoker, region):
    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }])
    cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes',
            'region_northwest', 'region_southeast', 'region_southwest']
    input_data = input_data[cols]
    input_data[['age', 'bmi', 'children']] = scaler.transform(input_data[['age', 'bmi', 'children']])
    return model.predict(input_data)[0]

# Placeholder for the green bar at the top
top_bar = st.empty()

# Input fields (compact layout)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age (years)", 18, 100, 30)
with col2:
    bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
with col3:
    children = st.slider("Children", 0, 5, 0)

col4, col5, col6 = st.columns(3)
with col4:
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
with col5:
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
with col6:
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Compute prediction
predicted_cost = predict(age, bmi, children, sex, smoker, region)

# Update the top green bar
top_bar.markdown(f"""
<div class="prediction-bar">
    <div class="prediction-label">Predicted Annual Medical Cost</div>
    <div class="prediction-value">${predicted_cost:,.0f}</div>
</div>
""", unsafe_allow_html=True)