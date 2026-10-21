# Loan Default Risk Prediction Using Machine Learning

## 1. Project Overview

This project develops an end-to-end machine learning system to estimate the risk that a loan applicant may default. Utilizing the **Home Credit Default Risk** dataset, the system processes financial characteristics, trains robust classification models, evaluates them using balanced metrics, handles interpretability via SHAP and LIME, and deploys predictions through an interactive Streamlit web application.

## 2. Objectives

- Explore, clean, and preprocess loan applicant data.
- Train and compare multiple classification models (Logistic Regression, Decision Trees, and Random Forest).
- Account for severe dataset class imbalance using balanced class weights.
- Implement global and local model explainability using SHAP and LIME.
- Serialize trained models and preprocessing pipelines using Joblib.
- Provide an interactive Streamlit user interface for real-time risk predictions.
- Document repository layout, local execution, and deployment steps.

## 3. Dataset & File Structure

The project directory contains the following files and components:

```text
loan-default-risk-prediction/
├── app.py                          # Streamlit interactive web application
├── application_train.csv           # Training dataset
├── application_test.csv            # Testing dataset
├── form_model.joblib               # Unified form prediction model pipeline
├── preprocessor.joblib             # Fitted data preprocessor pipeline
├── random_forest_model.joblib      # Serialized Random Forest model artifact
├── requirements.txt                # Python package dependencies
└── Untitled.ipynb                  # Development and modeling notebook
```

###4. Technologies Used
Python
pandas & NumPy
scikit-learn
Matplotlib & Seaborn
Joblib
Streamlit
SHAP & LIME
Jupyter Notebook

###5. Model Development & Evaluation Workflow

Data Preprocessing: Handled missing values via imputation, scaled numeric features, and encoded categorical variables using Scikit-Learn ColumnTransformer and Pipeline.
Handling Imbalance: Applied class_weight='balanced' across classifiers to ensure the model effectively learns patterns for the minority default class.
Model Evaluation: Evaluated models using Accuracy, Macro/Class-1 F1-scores, Confusion Matrices, and ROC-AUC to prevent the "accuracy trap" common in imbalanced datasets.
Serialization: Saved fitted models and preprocessors using joblib for seamless inference.

###6. Explainability

SHAP (SHapley Additive exPlanations): Used to compute global feature importances and interpret overall model behavior.
LIME (Local Interpretable Model-agnostic Explanations): Applied locally to explain individual applicant predictions and feature contributions transparently.

###7. Deployment
Deployment URL: https://loandefaultriskpredictions-he7jbnvnjoltywstwkstys.streamlit.app/


###8. Conclusion

This project demonstrates an end-to-end machine-learning workflow for
loan default risk prediction, from data preparation and model evaluation
to explainability and application delivery. The final conclusions should
be based on the measured evaluation results and the components that have
been successfully implemented and tested.

###9. Authors

-   Aby K Prakash CSD S3
-   Agnus Joshy   CSD S3
-   Akshaya A     CSD S3
-   Althea Maria  CSD S3

###10. References

-   Home Credit Default Risk dataset: https://github.com/agnusjoshy/loan_default_risk_predictions.git
-   scikit-learn documentation: https://scikit-learn.org/
-   Streamlit documentation:https://lnt-project-c7ge8buefjzs2czuy8cbdg.streamlit.app/
-   MLflow documentation: https://mlflow.org/docs/latest/
-   SHAP documentation: https://shap.readthedocs.io/
-   LIME documentation: https://lime-ml.readthedocs.io/

------------------------------------------------------------------------


