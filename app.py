import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessor
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("voting_model.joblib")

st.title("Loan Default Risk Prediction")
st.write("Enter applicant details to predict loan repayment risk.")

st.info(
    "Upload a CSV file containing applicant data with the same "
    "feature columns used during model training."
)

uploaded_file = st.file_uploader("Upload applicant CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Remove the target and ID columns if present
    data = data.drop(
        columns=["TARGET", "SK_ID_CURR"],
        errors="ignore"
    )

    try:
        processed_data = preprocessor.transform(data)
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]

        results = pd.DataFrame({
            "Prediction": [
                "High Risk" if p == 1 else "Low Risk"
                for p in predictions
            ],
            "Default Probability (%)": (
                probabilities * 100
            ).round(2)
        })

        st.subheader("Prediction Results")
        st.dataframe(results)

        st.download_button(
            "Download Results",
            data=results.to_csv(index=False),
            file_name="loan_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Could not process the CSV: {e}")
        st.write(
            "Please upload a CSV containing the expected applicant "
            "features with the same column names as the training data."
        )