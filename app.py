import streamlit as st
import pandas as pd
import joblib

from src.preprocess import preprocess, columns

# Load model
model = joblib.load("models/best_model.pkl")

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

st.title("🚨 AI Intrusion Detection System")
st.write("Upload network traffic data to detect cyber attacks")

# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv", "txt"])

if uploaded_file:
    try:
        # Read file with correct column names
        df = pd.read_csv(uploaded_file, names=columns)

        st.subheader("📊 Raw Data")
        st.dataframe(df.head())

        # Preprocess data
        df_processed = preprocess(df)

        # Predict
        predictions = model.predict(
            df_processed.drop("label", axis=1, errors="ignore")
        )

        df["Prediction"] = predictions

        # Map labels to readable form
        label_map = {
            0: "Normal",
            1: "DoS",
            2: "Probe",
            3: "R2L",
            4: "U2R",
            5: "Other"
        }

        df["Prediction"] = df["Prediction"].map(label_map)

        st.subheader("🔍 Prediction Results")
        st.dataframe(df.head())

        # Visualization
        st.subheader("📈 Attack Distribution")
        st.bar_chart(df["Prediction"].value_counts())

    except Exception as e:
        st.error(f"Error: {e}")