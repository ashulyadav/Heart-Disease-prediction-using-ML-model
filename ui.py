import streamlit as st

def app():
    st.title("🚀 Heart Disease Prediction App")
    st.write("Enter patient details below to predict heart disease risk.")

    # 👉 Input Fields
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

    # 👉 Show UI Always, but Store Input Only When Button is Clicked
    user_input = {}
    if st.button("Predict Heart Disease Risk"):
        user_input = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
    return user_input if user_input else None