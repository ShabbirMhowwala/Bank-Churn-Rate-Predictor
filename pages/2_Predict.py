"""Prediction page."""

from __future__ import annotations

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, section_title
from utils.app_data import explain_customer, risk_band, score_customer

import streamlit as st


page_bootstrap("Predict | FinGuard AI")
app_sidebar("Predict")

hero(
    "Customer Risk Scoring",
    (
        "Score a banking customer with the trained neural network. The result is not just a "
        "number: it includes confidence, risk category, and recommended retention actions "
        "that make the model useful in a real fintech workflow."
    ),
    "Live neural inference",
    "Enter customer details below",
)

section_title(
    "Customer profile",
    "Build a realistic banking scenario",
    "Adjust the customer attributes and submit the profile to generate a churn-risk probability.",
)

with st.form("premium_prediction_form"):
    c1, c2, c3 = st.columns(3)
    credit_score = c1.slider("Credit score", 300, 900, 650)
    geography = c2.selectbox("Country", ["France", "Germany", "Spain"])
    gender = c3.selectbox("Gender", ["Female", "Male"])

    c4, c5, c6 = st.columns(3)
    age = c4.slider("Age", 18, 92, 38)
    tenure = c5.slider("Bank tenure", 0, 10, 4)
    products = c6.select_slider("Products used", options=[1, 2, 3, 4], value=1)

    c7, c8 = st.columns(2)
    balance = c7.number_input("Account balance", min_value=0.0, value=85000.0, step=1000.0)
    salary = c8.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)

    c9, c10 = st.columns(2)
    has_card = c9.toggle("Has credit card", value=True)
    active = c10.toggle("Active member", value=True)

    submitted = st.form_submit_button("Generate AI Risk Score")

values = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": int(has_card),
    "IsActiveMember": int(active),
    "EstimatedSalary": salary,
}

if submitted:
    with st.spinner("Neural network is scoring the customer profile..."):
        probability = score_customer(values)
    label, css_class, recommendation = risk_band(probability)

    section_title("Prediction result", "AI-generated churn risk")
    r1, r2 = st.columns([0.9, 1.1])
    with r1:
        st.markdown(
            f"""
            <div class="glass-card">
                <p class="metric-label">Exit probability</p>
                <h3>{probability:.1%}</h3>
                <span class="risk-pill {css_class}">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))
    with r2:
        st.markdown(
            f"""
            <div class="glass-card">
                <p class="metric-label">Recommended action</p>
                <h3>{recommendation}</h3>
                <p>Use this output as a decision-support layer, not an automatic final decision. A real bank would add policy, fairness, and compliance checks before customer action.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title("Model explanation", "Signals behind the recommendation")
    for reason in explain_customer(values, probability):
        st.markdown(f"<div class='recommendation-card'>{reason}</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="glass-card">
            <h3>How to read the score</h3>
            <p>Low-risk customers are likely stable, watchlist customers deserve engagement nudges, and high-risk customers should receive fast retention attention. The model learns from historical bank churn patterns and returns a probability from 0 to 1.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

footer()
