from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH
from src.data import clean_telco_data, load_telco_data, split_features_target
from src.modeling import train_best_model


st.set_page_config(
    page_title="Telecom Churn Predictor",
    layout="wide",
)


@st.cache_resource(show_spinner="Training model for the demo...")
def get_model():
    if DEFAULT_MODEL_PATH.exists():
        return joblib.load(DEFAULT_MODEL_PATH)

    raw_df = load_telco_data(download=True)
    cleaned_df = clean_telco_data(raw_df)
    X, y = split_features_target(cleaned_df)
    result = train_best_model(X, y)

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, DEFAULT_MODEL_PATH)
    DEFAULT_METRICS_PATH.write_text(
        json.dumps(result.metrics, indent=2),
        encoding="utf-8",
    )
    return result.pipeline


def customer_from_form() -> pd.DataFrame:
    with st.form("customer_profile"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c2:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with c3:
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            )

        c4, c5 = st.columns(2)
        with c4:
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        with c5:
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(monthly_charges * max(tenure, 1)),
                step=10.0,
            )

        submitted = st.form_submit_button("Predict churn")

    row = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return submitted, pd.DataFrame([row])


def show_metrics() -> None:
    if not DEFAULT_METRICS_PATH.exists():
        return

    metrics = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        if key in metrics:
            col.metric(key.replace("_", " ").title(), metrics[key])


def score_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    scoring_df = df.copy()
    if "customerID" in scoring_df.columns:
        customer_ids = scoring_df["customerID"]
    else:
        customer_ids = pd.Series(range(1, len(scoring_df) + 1), name="customerID")

    if "Churn" in scoring_df.columns:
        scoring_df = scoring_df.drop(columns=["Churn"])

    scoring_df["TotalCharges"] = pd.to_numeric(scoring_df["TotalCharges"], errors="coerce")
    scoring_df["TotalCharges"] = scoring_df["TotalCharges"].fillna(
        scoring_df["MonthlyCharges"] * scoring_df["tenure"]
    )
    scoring_df["PaymentMethod"] = scoring_df["PaymentMethod"].str.replace(
        " (automatic)", "", regex=False
    )

    probabilities = model.predict_proba(scoring_df)[:, 1]
    labels = ["Yes" if value >= 0.5 else "No" for value in probabilities]

    return pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_probability": probabilities.round(3),
            "predicted_churn": labels,
        }
    ).sort_values("churn_probability", ascending=False)


st.title("Telecom Customer Churn Predictor")
st.caption("Score customer churn risk using a reproducible scikit-learn pipeline.")

model = get_model()
show_metrics()

tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring"])

with tab_single:
    submitted, customer_df = customer_from_form()
    if submitted:
        probability = model.predict_proba(customer_df)[0, 1]
        prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"

        left, right = st.columns([1, 2])
        left.metric("Churn probability", f"{probability:.1%}")
        right.progress(float(probability))
        st.subheader(prediction)

with tab_batch:
    uploaded = st.file_uploader("Upload a Telco customer CSV", type=["csv"])
    if uploaded is not None:
        upload_df = pd.read_csv(uploaded)
        predictions = score_dataframe(model, upload_df)
        st.dataframe(predictions, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a CSV with the same customer columns as the Telco churn dataset.")
