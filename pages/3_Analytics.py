"""Analytics page."""

from __future__ import annotations

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, section_title
from utils.app_data import (
    churn_by_category,
    fig_age_distribution,
    fig_balance_salary,
    fig_churn_by_geo,
    fig_product_churn,
    load_dataset,
)

import plotly.express as px
import streamlit as st


page_bootstrap("Analytics | FinGuard AI")
app_sidebar("Analytics")

df = load_dataset()

hero(
    "Retention Analytics Lab",
    (
        "Explore how churn changes across countries, products, activity levels, balances, "
        "and customer age. This page makes the model context visible so recruiters can see "
        "that the project is more than a black-box prediction form."
    ),
    "Portfolio analytics",
)

section_title("Filters", "Slice the customer portfolio")
countries = st.multiselect("Countries", sorted(df["Geography"].unique()), default=sorted(df["Geography"].unique()))
activity = st.multiselect("Activity status", ["Active", "Inactive"], default=["Active", "Inactive"])

filtered = df[df["Geography"].isin(countries)]
activity_map = {"Active": 1, "Inactive": 0}
filtered = filtered[filtered["IsActiveMember"].isin([activity_map[x] for x in activity])]

section_title(
    "Segmentation",
    "Churn patterns across customer groups",
    "Interactive charts help a growth team find practical retention opportunities.",
)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig_churn_by_geo(filtered), use_container_width=True)
with c2:
    st.plotly_chart(fig_product_churn(filtered), use_container_width=True)

st.plotly_chart(fig_balance_salary(filtered), use_container_width=True)
st.plotly_chart(fig_age_distribution(filtered), use_container_width=True)

section_title("Retention table", "Highest-risk groups")
summary = churn_by_category(filtered, "Geography")
st.dataframe(
    summary.assign(churn_rate=summary["churn_rate"].map(lambda x: f"{x:.1%}")),
    use_container_width=True,
    hide_index=True,
)

footer()
