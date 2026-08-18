import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Pay Equity Checker", layout="centered")

@st.cache_resource
def load_artifact():
    return joblib.load("salary_model.joblib")

artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
threshold = artifact["deviation_threshold_pct"]

st.title("Pay Equity Checker")
st.caption(f"Model: {artifact['model_name']}")
st.write(
    "Enter an employee's job-related details (not gender -- the model is deliberately "
    "gender-blind) to see their model-predicted 'fair' pay, then compare it to their actual pay."
)

with st.form("employee_form"):
    job_title = st.text_input("Job Title", "Data Scientist")
    age = st.number_input("Age", 18, 80, 34)
    perf_eval = st.slider("Performance Evaluation (1-5)", 1, 5, 3)
    education = st.selectbox("Education", ["High School", "College", "Masters", "PhD"])
    dept = st.text_input("Department", "Engineering")
    seniority = st.slider("Seniority (years)", 1, 10, 3)
    actual_pay = st.number_input("Actual Base Pay ($)", 0, 500000, 90000)

    submitted = st.form_submit_button("Check Pay Fairness")

if submitted:
    raw = pd.DataFrame([{
        "JobTitle": job_title, "Age": age, "PerfEval": perf_eval,
        "Education": education, "Dept": dept, "Seniority": seniority,
    }])
    encoded = pd.get_dummies(raw)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(encoded)

    predicted = model.predict(scaled)[0]
    deviation_pct = (actual_pay - predicted) / predicted * 100

    if deviation_pct <= -threshold:
        flag, color = "Underpaid", "red"
    elif deviation_pct >= threshold:
        flag, color = "Overpaid", "orange"
    else:
        flag, color = "Fair", "green"

    st.metric("Model-predicted fair pay", f"${predicted:,.0f}")
    st.metric("Deviation from predicted", f"{deviation_pct:+.1f}%")
    st.markdown(f"**Flag:** :{color}[{flag}]")
    st.caption(
        "This model never sees gender as an input -- it predicts pay from job factors only. "
        "A consistent pattern of one group being flagged Underpaid more often is what a real "
        "pay-equity audit would investigate further."
    )
