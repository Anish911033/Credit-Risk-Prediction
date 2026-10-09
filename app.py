import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Risk", page_icon="💳")

# Same order/encoding as the notebook
GRADE_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_percent_income", "cb_person_cred_hist_length",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


model, scaler, feature_columns = load_artifacts()

st.title("💳 Loan Default Risk Prediction")
st.write("Enter the applicant's details to predict whether the loan is high-risk or low-risk.")

with st.form("applicant_form"):
    st.subheader("Applicant")
    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual income (USD)", 4000, 300000, 60000, step=1000)
    person_emp_length = st.number_input("Employment length (years)", 0.0, 50.0, 5.0, step=0.5)

    st.subheader("Home & loan")
    person_home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = st.selectbox(
        "Loan intent",
        ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"],
    )
    loan_grade = st.selectbox("Loan grade", list(GRADE_ORDER))
    loan_amnt = st.number_input("Loan amount (USD)", 500, 35000, 10000, step=500)

    st.subheader("Credit history")
    cb_default = st.selectbox("Previously defaulted?", ["N", "Y"])
    cb_person_cred_hist_length = st.number_input("Credit history length (years)", 2, 30, 5)

    submitted = st.form_submit_button("Predict risk")

if submitted:
    # Notebook recomputes this from loan amount and income, so we do too
    loan_percent_income = round(loan_amnt / person_income, 2)

    row = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_grade": GRADE_ORDER[loan_grade],
        "loan_amnt": loan_amnt,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": 1 if cb_default == "Y" else 0,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
    }])

    # One-hot without drop_first (a single row would lose its only category);
    # reindexing to the training columns drops the baseline categories
    # (MORTGAGE, DEBTCONSOLIDATION) and fills any missing dummy with 0.
    row = pd.get_dummies(row, columns=["person_home_ownership", "loan_intent"], dtype=int)
    row = row.reindex(columns=feature_columns, fill_value=0)
    row[NUMERIC_COLS] = scaler.transform(row[NUMERIC_COLS])

    default_idx = list(model.classes_).index(1)
    p_default = model.predict_proba(row)[0][default_idx]

    st.subheader("Result")
    if p_default > 0.5:
        st.error("🔴 HIGH-RISK: likely to default")
    else:
        st.success("🟢 LOW-RISK: unlikely to default")
    st.metric("Probability of default", f"{p_default:.1%}")
    st.progress(float(p_default))
    st.caption(f"Loan is {loan_percent_income:.0%} of annual income.")
    if loan_percent_income > 1:
        st.warning("Loan exceeds annual income, which is outside the range the model was trained on.")

st.caption("Random Forest (200 trees) trained on the Kaggle Credit Risk dataset. Educational project, not for real lending decisions.")
