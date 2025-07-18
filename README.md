# 💵 Currency Classifier using BankNote-Net + Streamlit

This project is a Streamlit-based web application that classifies currency notes (e.g., INR, USD, EUR) using vector embeddings from the [BankNote-Net](https://github.com/microsoft/BanknoteNet) dataset. It leverages a Logistic Regression model trained on 256-dimensional feature vectors with over 95% accuracy across 17 global currencies.

---

## 🚀 Demo

Upload a currency embedding (CSV with 256 features) and get an instant prediction:

> Predicted Currency: **INR** 🇮🇳  
> Predicted Currency: **USD** 🇺🇸  
> Predicted Currency: **EUR** 🇪🇺

## 🧠 Features

- ✔️ Pretrained model on 24k+ currency images from 17 countries
- ✔️ Logistic Regression with 95% accuracy
- ✔️ User-friendly Streamlit interface
- ✔️ Upload `.csv` files containing note embeddings
- ✔️ Supports all major currencies: INR, USD, EUR, JPY, GBP, etc.

## 📂 Project Structure
currency_classifier/
├── app.py ← Streamlit Web App
├── currency_model.pkl ← Trained ML Model
├── label_encoder.pkl ← Label decoder
├── test_inr.csv ← Sample test input (optional)
└── README.md

## 📈 Model Details

- Dataset: [BankNote-Net](https://github.com/microsoft/BanknoteNet)
- Features: 256-D embedding vectors (v₀ to v₂₅₅)
- Target: `Currency` column (17 unique values)
- Model: Logistic Regression (`sklearn`)
- Accuracy: **95.1%**

## 💻 How to Run (Local)

1. Clone this repo and install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn


2. Launch the app:
```bash
streamlit run app.py
