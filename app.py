"""
Churn Prediction Dashboard (ANN) — pure Streamlit, no HTML/CSS/JS.

Run with:  streamlit run app.py
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="wide")


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
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    df = pd.read_csv("churn_clean.csv")
    return model, scaler, encoders, metrics, df


model, scaler, encoders, metrics, df = load_artifacts()
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
# SIDEBAR NAV
# ---------------------------------------------------------------------------
st.sidebar.title("📉 Churn Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Explore Data", "Model Performance", "Predict Churn"])

# ---------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Customer Churn Prediction — ANN Dashboard")
    st.caption("Bank customer churn dataset · Artificial Neural Network classifier")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{metrics['churn_rate']*100:.1f}%")
    c3.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Geography")
        geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        st.bar_chart(geo_churn)

    with col2:
        st.subheader("Churn by Gender")
        gender_churn = df.groupby("Gender")["Exited"].mean()
        st.bar_chart(gender_churn)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Churn by Number of Products")
        prod_churn = df.groupby("NumOfProducts")["Exited"].mean()
        st.bar_chart(prod_churn)
    with col4:
        st.subheader("Churn by Active Membership")
        active_churn = df.groupby("IsActiveMember")["Exited"].mean()
        active_churn.index = ["Inactive", "Active"]
        st.bar_chart(active_churn)

# ---------------------------------------------------------------------------
# PAGE 2: EXPLORE DATA
# ---------------------------------------------------------------------------
elif page == "Explore Data":
    st.title("Explore the Dataset")

    with st.expander("Raw data sample"):
        st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Numeric feature distributions")
    numeric_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure"]
    col = st.selectbox("Choose a feature", numeric_cols)
    hist_data = df[[col, "Exited"]].copy()
    hist_data["Exited"] = hist_data["Exited"].map({0: "Stayed", 1: "Churned"})
    binned = hist_data.groupby(pd.cut(hist_data[col], bins=15))["Exited"] \
        .apply(lambda s: (s == "Churned").mean())
    binned.index = binned.index.astype(str)  # Altair can't render Interval objects
    st.bar_chart(binned)

    st.subheader("Correlation with churn (numeric features)")
    corr = df.select_dtypes(include=[np.number]).corr()["Exited"].drop("Exited").sort_values()
    st.bar_chart(corr)

# ---------------------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("ANN Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c3.metric("Test Samples", metrics["n_test"])

    st.subheader("Training History")
    hist_df = pd.DataFrame({
        "train_loss": metrics["history"]["loss"],
        "val_loss": metrics["history"]["val_loss"],
    })
    st.line_chart(hist_df)

    hist_acc_df = pd.DataFrame({
        "train_accuracy": metrics["history"]["accuracy"],
        "val_accuracy": metrics["history"]["val_accuracy"],
    })
    st.line_chart(hist_acc_df)

    st.subheader("Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Actual: Stayed", "Actual: Churned"],
                          columns=["Pred: Stayed", "Pred: Churned"])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("ROC Curve")
    roc_df = pd.DataFrame({
        "FPR": metrics["roc_curve"]["fpr"],
        "TPR": metrics["roc_curve"]["tpr"],
    }).set_index("FPR")
    st.line_chart(roc_df)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report_df.round(3), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 4: PREDICT CHURN (live inference form)
# ---------------------------------------------------------------------------
elif page == "Predict Churn":
    st.title("Predict Churn for a Customer")
    st.caption("Fill in the customer's details and the trained ANN will estimate churn probability.")

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
