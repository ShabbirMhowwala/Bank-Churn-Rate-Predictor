"""Dashboard page."""

from __future__ import annotations

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, metric_card, section_title
from utils.app_data import fig_age_distribution, fig_churn_by_geo, fig_product_churn, load_dataset, load_metrics

import streamlit as st


page_bootstrap("Dashboard | FinGuard AI")
app_sidebar("Dashboard")

df = load_dataset()
metrics = load_metrics()

hero(
    "AI Retention Command Center",
    (
        "A startup-style cockpit for customer churn intelligence. FinGuard AI combines "
        "deep learning predictions with business-friendly explanations, helping fintech "
        "teams prioritize customers before revenue walks out the door."
    ),
    "Dashboard",
    "Monitor portfolio risk in real time",
)

cols = st.columns(4)
with cols[0]:
    metric_card("Total Customers", f"{len(df):,}", "Dataset coverage")
with cols[1]:
    metric_card("Exited", f"{df['Exited'].sum():,}", "Known churn cases", "violet")
with cols[2]:
    metric_card("Model AUC", f"{metrics.get('roc_auc', 0):.3f}", "Ranking quality", "cyan")
with cols[3]:
    metric_card("Avg Balance", f"{df['Balance'].mean():,.0f}", "Customer value signal", "violet")

section_title(
    "Strategic overview",
    "From raw bank records to retention decisions",
    (
        "The original dataset contains credit score, geography, gender, age, tenure, account "
        "balance, product count, credit-card status, active membership, salary, and exit label. "
        "The app transforms those signals into a neural churn-risk score and pairs it with "
        "charts a product or growth team can understand quickly."
    ),
)

left, right = st.columns(2)
with left:
    st.plotly_chart(fig_churn_by_geo(df), use_container_width=True)
with right:
    st.plotly_chart(fig_product_churn(df), use_container_width=True)

section_title(
    "Customer age profile",
    "Age bands reveal retention pressure",
    "Use this view to see whether churn is concentrated among younger digital-first users or older high-value customers.",
)
st.plotly_chart(fig_age_distribution(df), use_container_width=True)

footer()
