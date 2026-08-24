from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "extra_trees_credit_model.pkl")
encoder = {
    column: joblib.load(BASE_DIR / f"{column}_encoder.pkl")
    for column in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_accounts = st.selectbox(
    "Saving Accounts", ["little", "moderate", "rich", "quite rich"]
)
checking_account = st.selectbox(
    "Checking Account", ["little", "moderate", "rich"]
)
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame(
    {
        "Age": [age],
        "Sex": [encoder["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoder["Housing"].transform([housing])[0]],
        "Saving accounts": [
            encoder["Saving accounts"].transform([saving_accounts])[0]
        ],
        "Checking account": [
            encoder["Checking account"].transform([checking_account])[0]
        ],
        "Credit amount": [credit_amount],
        "Duration": [duration],
    }
)

if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    risk_label = "GOOD" if prediction == 1 else "BAD"

    if risk_label == "GOOD":
        st.success(f"The predicted risk is: **{risk_label}**")
    else:
        st.error(f"The predicted risk is: **{risk_label}**")
