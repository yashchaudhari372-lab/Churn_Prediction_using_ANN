"""
Churn Prediction Dashboard (ANN) — pure Streamlit, no HTML/CSS/JS.

Run with:  streamlit run app.py
"""

import pickle

import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="centered")


# ---------------------------------------------------------------------------
# LOAD ARTIFACTS (cached so they only load once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_ann_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


model, scaler, encoders = load_artifacts()
le_gender = encoders["le_gender"]
ohe_geo = encoders["ohe_geo"]
FEATURE_COLUMNS = encoders["feature_columns"]
GEO_CATEGORIES = encoders["geo_categories"]


def preprocess_single(record: dict) -> np.ndarray:
    """Turn a raw form input dict into a scaled feature vector, in the exact
    column order the model was trained on."""
    row = {
        "CreditScore": record["CreditScore"],
        "Gender": le_gender.transform([record["Gender"]])[0],
        "Age": record["Age"],
        "Tenure": record["Tenure"],
        "Balance": record["Balance"],
        "NumOfProducts": record["NumOfProducts"],
        "HasCrCard": record["HasCrCard"],
        "IsActiveMember": record["IsActiveMember"],
        "EstimatedSalary": record["EstimatedSalary"],
    }
    geo_encoded = ohe_geo.transform([[record["Geography"]]])[0]
    for i, cat in enumerate(GEO_CATEGORIES[1:]):
        row[f"Geography_{cat}"] = geo_encoded[i]

    ordered = np.array([[row[c] for c in FEATURE_COLUMNS]])
    return scaler.transform(ordered)


# ---------------------------------------------------------------------------
# DASHBOARD (single page: live prediction)
# ---------------------------------------------------------------------------
st.title("📉 Churn Prediction Dashboard")
st.caption("Enter a customer's details and the trained ANN will estimate their churn probability.")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", GEO_CATEGORIES)
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0, step=1000.0)
    with col3:
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0, step=1000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }
    X = preprocess_single(record)
    prob = float(model.predict(X, verbose=0)[0][0])

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        if prob >= 0.5:
            st.error("⚠️ High risk of churn")
        else:
            st.success("✅ Likely to stay")
    with c2:
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Model decision threshold: 50%. Raw probability: {prob:.4f}")
