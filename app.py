"""FinGuard AI premium Streamlit entry point."""

from __future__ import annotations

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, metric_card, section_title
from utils.app_data import fig_balance_salary, fig_churn_by_geo, fig_product_churn, load_dataset, load_metrics

import streamlit as st


def main() -> None:
    page_bootstrap("FinGuard AI | Customer Churn Intelligence")
    app_sidebar("Dashboard")

    df = load_dataset()
    metrics = load_metrics()

    hero(
        title="FinGuard AI",
        subtitle=(
            "A premium deep-learning churn intelligence platform for digital banks, "
            "neobanks, and fintech retention teams. It turns customer profile and "
            "banking behavior signals into clear exit-risk decisions."
        ),
        badge="Neural customer retention engine",
        primary="Open Predict page to score a customer",
    )

    section_title(
        "Executive cockpit",
        "Banking churn risk at a glance",
        (
            "This dashboard summarizes the customer base, highlights high-risk segments, "
            "and provides a clean product story for interviews: data ingestion, feature "
            "engineering, neural-network scoring, evaluation, and a polished business UI."
        ),
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Customers", f"{len(df):,}", "Historical banking profiles")
    with c2:
        metric_card("Churn Rate", f"{df['Exited'].mean():.1%}", "Observed exit behavior", "violet")
    with c3:
        metric_card("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}", "ANN separation power", "cyan")
    with c4:
        metric_card("Recall", f"{metrics.get('recall', 0):.3f}", "Risk customers captured", "violet")

    section_title(
        "Insight layer",
        "Where customer risk concentrates",
        (
            "Churn rarely appears randomly. The app surfaces country-level, product-level, "
            "balance, age, and activity patterns so a fintech team can decide where to focus "
            "retention offers and relationship-manager outreach."
        ),
    )

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(fig_churn_by_geo(df), use_container_width=True)
    with g2:
        st.plotly_chart(fig_product_churn(df), use_container_width=True)

    section_title("Behavior map", "Salary, balance, and churn movement")
    st.plotly_chart(fig_balance_salary(df), use_container_width=True)

    footer()


if __name__ == "__main__":
    main()
