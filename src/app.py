import streamlit as st
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"

st.title("Heart Disease Risk Predictor")
st.write("Enter patient details to estimate heart disease risk.")
st.caption("This demo uses the trained heart-disease model saved in the project. ")

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 500, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    record = {
        "age": age, "sex": 1 if sex == "Male" else 0, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal,
    }
    row = pd.DataFrame([record])
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    proba = model.predict_proba(row_scaled)[0][1]

    if prediction == 1:
        st.error(f"High risk of heart disease ({proba*100:.1f}% probability)")
    else:
        st.success(f"Low risk of heart disease ({proba*100:.1f}% probability)")