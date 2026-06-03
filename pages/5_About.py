"""About page."""

from __future__ import annotations

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, lottie_pulse, section_title

import streamlit as st


page_bootstrap("About | FinGuard AI")
app_sidebar("About")

hero(
    "About FinGuard AI",
    (
        "FinGuard AI is a portfolio-ready fintech machine-learning product built around a "
        "deep neural network for bank customer churn prediction. The goal is to show clean "
        "ML engineering, business relevance, and a polished Streamlit product experience."
    ),
    "Project story",
)

section_title("Problem", "Why churn prediction matters in fintech")
about_left, about_right = st.columns([1.4, 0.6])
with about_left:
    st.markdown(
        """
        <div class="glass-card">
            <p>Customer acquisition is expensive for banks and fintech apps. When a customer exits, the business loses future deposits, product usage, cross-sell potential, and relationship data. A churn intelligence system helps teams identify risky accounts earlier and design retention actions that feel timely rather than desperate.</p>
            <p>This project frames churn prediction as a practical AI product: a model that can be trained locally, evaluated clearly, and used through a polished dashboard by non-technical teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with about_right:
    lottie_pulse(180)

section_title("ML workflow", "How the system works")
cols = st.columns(4)
items = [
    ("1", "Load customer data", "Credit score, demographics, balance, products, salary, activity."),
    ("2", "Engineer features", "Balance ratios, tenure-age behavior, high-value inactive flag."),
    ("3", "Train ANN", "Keras dense network with dropout, batch normalization, and sigmoid output."),
    ("4", "Deploy dashboard", "Streamlit UI with live scoring, explanations, and analytics."),
]
for col, (step, title, body) in zip(cols, items):
    with col:
        st.markdown(
            f"<div class='glass-card'><p class='eyebrow'>Step {step}</p><h3>{title}</h3><p>{body}</p></div>",
            unsafe_allow_html=True,
        )

section_title("Tech stack", "Production-style but beginner friendly")
st.markdown(
    """
    <div class="glass-card">
        <p><b>Python</b>, <b>TensorFlow/Keras</b>, <b>Scikit-learn</b>, <b>Pandas</b>, <b>NumPy</b>, <b>Plotly</b>, and <b>Streamlit</b>. The backend remains simple enough to explain in interviews while the frontend looks like a serious AI SaaS prototype.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

footer()
