import streamlit as st
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the trained model
model = joblib.load('temperature_predictor.pkl')

# Streamlit app
st.title("Temperature Prediction App")

st.header("Enter the input parameters:")
# Input fields for user to provide input
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
light_intensity = st.number_input("Light Intensity (lux)", min_value=0.0, max_value=10000.0, step=0.1)
time_category_numeric = st.number_input("Time [0 for Morning, 1 for Afternoon, or 2 for Evening]", min_value=0, max_value=3, step=1)
location_category = st.number_input("Location [0 for Indoor, 1 for Outdoor]", min_value=0, max_value=1, step=1)

# Predict temperature
if st.button("Predict Temperature"):
    input_data = np.array([[humidity, light_intensity, time_category_numeric, location_category]])
    predicted_temp = model.predict(input_data)
    st.success(f"Predicted Temperature: {predicted_temp[0]:.2f}°C")
