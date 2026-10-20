import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained form model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /  "form_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the prediction model: {e}")
    st.stop()

# Website title
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Default Risk Prediction")
st.write(
    "Enter the applicant's financial details "
    "to estimate loan default risk."
)

# Applicant input form
with st.form("loan_form"):

    st.subheader("Applicant Details")

    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=300000.0,
        step=10000.0
    )

    credit = st.number_input(
        "Loan Credit Amount",
        min_value=0.0,
        value=500000.0,
        step=10000.0
    )

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=30
    )

    employment = st.number_input(
        "Employment Experience (years)",
        min_value=0.0,
        max_value=80.0,
        value=5.0,
        step=1.0
    )

    family = st.number_input(
        "Number of Family Members",
        min_value=1,
        max_value=30,
        value=3
    )

    annuity = st.number_input(
        "Annual Loan Payment",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )

    submitted = st.form_submit_button("Predict Loan Risk")

# Prediction
if submitted:

    # Convert form values into the model's expected features
    input_data = pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "DAYS_BIRTH": -age * 365,
        "DAYS_EMPLOYED": -employment * 365,
        "CNT_FAM_MEMBERS": family,
        "AMT_ANNUITY": annuity
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error("Higher Estimated Default Risk")
    else:
        st.success("Lower Estimated Default Risk")

    st.metric(
        "Estimated Default Probability",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This is an educational machine-learning estimate, "
        "not a guarantee of loan repayment or a real lending decision."
    )
