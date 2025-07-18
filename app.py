import streamlit as st
import pickle
import numpy as np

# Load trained model and label encoder
with open('currency_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Title
st.title("💵 Currency Classifier (BankNote-Net)")
st.write("Upload a currency note embedding (CSV with 256 features) to identify the currency type.")

# Upload input file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    import pandas as pd

    try:
        data = pd.read_csv(uploaded_file)

        # Check shape and predict
        if data.shape[1] == 256:
            prediction = model.predict(data.values)
            currency = le.inverse_transform(prediction)

            st.success(f"Predicted Currency: **{currency[0]}**")
        else:
            st.error("Input file must contain exactly 256 columns (v_0 to v_255).")

    except Exception as e:
        st.error(f"Error processing file: {e}")
