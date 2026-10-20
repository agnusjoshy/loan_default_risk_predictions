import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Get the project folder
BASE_DIR = Path(Loan_Default_Risk_Predictions).parent

# Load the trained model and preprocessor
MODEL_PATH = BASE_DIR / "voting_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.joblib"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

# Application title
st.title("🏦 Loan Default Risk Prediction")

st.caption(
    "Machine Learning Project using the Home Credit Default Risk dataset"
)

st.write(
    "Upload applicant information to predict the possibility "
    "of loan repayment difficulties."
)

st.info(
    "Prediction 0: No repayment difficulties predicted\n\n"
    "Prediction 1: Repayment difficulties predicted"
)

# Upload applicant data
uploaded_file = st.file_uploader(
    "Upload applicant data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        # Remove target and applicant ID if present
        data = data.drop(
            columns=["TARGET", "SK_ID_CURR"],
            errors="ignore"
        )

        # Get the original feature names
        if hasattr(preprocessor, "feature_names_in_"):
            expected_features = list(preprocessor.feature_names_in_)

            # Check for missing columns
            missing_features = [
                col for col in expected_features
                if col not in data.columns
            ]

            if missing_features:
                st.error(
                    "The uploaded CSV is missing "
                    f"{len(missing_features)} required columns."
                )

                st.write("Missing columns:", missing_features)
                st.stop()

            # Keep the required columns in the correct order
            data = data[expected_features]

        st.success(
            f"Successfully loaded {len(data)} applicant record(s)."
        )

        # Select an applicant
        if len(data) > 0:
            applicant_index = st.selectbox(
                "Select applicant record",
                options=list(range(len(data))),
                format_func=lambda x: f"Applicant record {x + 1}"
            )

            if st.button("Predict Loan Default Risk"):

                # Select one applicant
                applicant = data.iloc[[applicant_index]]

                # Apply the fitted preprocessing steps
                processed_data = preprocessor.transform(applicant)

                # Make prediction
                prediction = model.predict(processed_data)[0]

                # Display result
                if prediction == 1:
                    st.error(
                        "⚠️ Repayment Difficulties Predicted"
                    )
                    st.write(
                        "The model predicts that this applicant "
                        "may experience repayment difficulties."
                    )
                else:
                    st.success(
                        "✅ No Repayment Difficulties Predicted"
                    )
                    st.write(
                        "The model predicts no repayment "
                        "difficulties for this applicant."
                    )

                # Display probabilities when supported
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        processed_data
                    )[0]

                    probability_df = pd.DataFrame({
                        "Prediction Class": model.classes_,
                        "Probability": [
                            f"{probability:.2%}"
                            for probability in probabilities
                        ]
                    })

                    st.subheader("Prediction Probabilities")
                    st.table(probability_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.divider()

st.caption(
    "Educational demonstration only. Predictions are not "
    "guarantees of an applicant's ability to repay a loan."
)
