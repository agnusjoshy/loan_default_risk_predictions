import joblib
import pandas as pd
import streamlit as st

# Load the single unified model pipeline
model = joblib.load("form_model.joblib")

st.title("Loan Default Risk Prediction")
st.write(
    "Enter the applicant's financial details below to predict loan repayment"
    " risk."
)

# Create input widgets for the specific fields
amt_income_total = st.number_input(
    "Total Income (AMT_INCOME_TOTAL)", min_value=0.0, value=50000.0, step=1000.0
)
amt_credit = st.number_input(
    "Credit Amount (AMT_CREDIT)", min_value=0.0, value=150000.0, step=5000.0
)
days_birth = st.number_input(
    "Age in Days (DAYS_BIRTH - negative number)",
    value=-12000,
    step=100,
    help="Ex: -12000 days roughly corresponds to ~32 years old.",
)
days_employed = st.number_input(
    "Employment Days (DAYS_EMPLOYED - negative number)",
    value=-2000,
    step=100,
    help="Days employed before the application.",
)
cnt_fam_members = st.number_input(
    "Family Members (CNT_FAM_MEMBERS)", min_value=1, value=2, step=1
)
amt_annuity = st.number_input(
    "Loan Annuity (AMT_ANNUITY)", min_value=0.0, value=10000.0, step=500.0
)

# Button to trigger prediction
if st.button("Predict Default Risk"):
  # Construct a DataFrame matching the exact feature order
  input_data = pd.DataFrame(
      {
          "AMT_INCOME_TOTAL": [amt_income_total],
          "AMT_CREDIT": [amt_credit],
          "DAYS_BIRTH": [days_birth],
          "DAYS_EMPLOYED": [days_employed],
          "CNT_FAM_MEMBERS": [cnt_fam_members],
          "AMT_ANNUITY": [amt_annuity],
      }
  )

  try:
    # Since the pipeline includes imputation and scaling, pass raw input directly
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= 0.60 else 0  # <--- Custom threshold

    risk_label = "High Risk (Default)" if prediction == 1 else "Low Risk (Safe)"

    st.subheader("Prediction Result")
    if prediction == 1:
      st.error(f"**Result:** {risk_label}")
    else:
      st.success(f"**Result:** {risk_label}")

    st.write(f"**Default Probability:** {probability * 100:.2f}%")

  except Exception as e:
    st.error(f"Could not process the prediction: {e}")import streamlit as st
