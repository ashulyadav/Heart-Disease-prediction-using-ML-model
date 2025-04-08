import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
import os
import sys

# Load the trained neural network model
MODEL_PATH = "heart_disease_model.keras" 
model = load_model(r"C:\Users\DELL\Desktop\codes\project\heartdisease\heart_disease_model.keras")

def predict_heart_disease(features):
    """Predict heart disease status using the neural network model."""
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return "High Risk" if prediction[0][0] > 0.5 else "Low Risk"

# Streamlit UI
def main():
    st.title("Heart Disease Prediction")
    st.write("Enter the patient details below to predict heart disease risk.")
    
    # Input fields
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 1)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    # Encoding categorical inputs
    sex = 1 if sex == "Male" else 0
    cp = ["Type 1", "Type 2", "Type 3", "Type 4"].index(cp)
    fbs = 1 if fbs == "Yes" else 0
    restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang = 1 if exang == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
    
    # Prediction button
    if st.button("Predict Heart Disease Risk"):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_heart_disease(features)
        st.subheader(f"Prediction: {result}")
        
st.set_page_config(page_title="Heart Disease Prediction")  # Set title

st.title("Heart Disease Prediction App")
st.write("Upload data to predict heart disease status.")

if __name__ == "__main__":
    import os
    os.system("streamlit run main.py --server.runOnSave true --server.enableCORS true")
#install latest version of tensorflow and keras 