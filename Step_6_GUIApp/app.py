import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL + ARTIFACTS
# -----------------------------
model = joblib.load("final_disease_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Disease Prediction System", layout="wide")

st.title("🩺 Multi-Disease Prediction System")
st.markdown("### Enter patient details to predict disease probabilities")

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 100.0, 220.0)
    weight = st.number_input("Weight (kg)", 30.0, 150.0)

with col2:
    systolic = st.number_input("Systolic BP", 80, 200)
    diastolic = st.number_input("Diastolic BP", 50, 130)
    cholesterol = st.number_input("Cholesterol", 100, 350)
    sugar = st.number_input("Fasting Sugar", 70, 200)

with col3:
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol = st.selectbox("Alcohol", [0, 1])
    activity = st.selectbox("Activity Level", [0, 1, 2])  # Low=0, Moderate=1, High=2
    steps = st.number_input("Daily Steps", 0, 20000)

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_input():
    bmi = weight / ((height / 100) ** 2)

    data = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Height_cm": height,
        "Weight_kg": weight,
        "BMI": bmi,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Cholesterol": cholesterol,
        "Fasting_Sugar": sugar,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Activity_Level": activity,
        "Daily_Steps": steps
    }

    df = pd.DataFrame([data])

    # -----------------------------
    # FEATURE ENGINEERING (MUST MATCH TRAINING)
    # -----------------------------
    df["Pulse_Pressure"] = df["Systolic_BP"] - df["Diastolic_BP"]
    df["MAP"] = (df["Systolic_BP"] + 2 * df["Diastolic_BP"]) / 3

    df["Lifestyle_Risk"] = (
        df["Smoking"] * 2 +
        df["Alcohol"] +
        (df["Activity_Level"] == 0).astype(int) * 2
    )

    # -----------------------------
    # ALIGN FEATURE ORDER
    # -----------------------------
    df = df.reindex(columns=feature_cols, fill_value=0)

    # -----------------------------
    # SCALE INPUT
    # -----------------------------
    df_scaled = scaler.transform(df)

    return df_scaled


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict"):
    try:
        processed = preprocess_input()
        preds = model.predict(processed)[0]

        diseases = [
            "Diabetes",
            "Hypertension",
            "Heart Disease",
            "Anemia",
            "Kidney Disease"
        ]

        st.markdown("## 📊 Prediction Results")

        for disease, prob in zip(diseases, preds):
            percentage = round(prob * 100, 2)

            # Risk color coding
            if percentage < 30:
                color = "green"
                label = "Low Risk"
            elif percentage < 70:
                color = "orange"
                label = "Moderate Risk"
            else:
                color = "red"
                label = "High Risk"

            st.markdown(
                f"### {disease}\n"
                f"<span style='color:{color}; font-size:22px'><b>{percentage}% ({label})</b></span>",
                unsafe_allow_html=True
            )

            st.progress(int(percentage))

    except Exception as e:
        st.error(f"❌ Error: {e}")

#run : python -m streamlit run app.py