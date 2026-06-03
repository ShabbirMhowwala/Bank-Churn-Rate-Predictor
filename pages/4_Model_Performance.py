"""Model performance page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.theme import page_bootstrap
from components.ui import app_sidebar, footer, hero, metric_card, section_title
from utils.app_data import ROOT_DIR, fig_feature_signal, fig_roc_curve, load_dataset, load_metrics, load_predictions


page_bootstrap("Model Performance | FinGuard AI")
app_sidebar("Model Performance")

df = load_dataset()
metrics = load_metrics()
preds = load_predictions()

hero(
    "Neural Network Performance",
    (
        "A transparent model room for interview conversations. It shows how the Keras ANN "
        "performed, where it makes mistakes, and which customer signals are most associated "
        "with churn in the dataset."
    ),
    "Evaluation layer",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Accuracy", f"{metrics.get('accuracy', 0):.3f}", "Overall correctness")
with c2:
    metric_card("Precision", f"{metrics.get('precision', 0):.3f}", "Predicted churn quality", "violet")
with c3:
    metric_card("Recall", f"{metrics.get('recall', 0):.3f}", "Churn cases captured")
with c4:
    metric_card("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}", "Ranking strength", "violet")

section_title(
    "ROC curve",
    "How well the model ranks churn risk",
    "A higher curve means the model separates exiting customers from retained customers more reliably.",
)
st.plotly_chart(fig_roc_curve(), use_container_width=True)

section_title("Training artifacts", "Saved evaluation visuals")
left, right = st.columns(2)
history_path = ROOT_DIR / "assets" / "training_history.png"
matrix_path = ROOT_DIR / "assets" / "confusion_matrix.png"
if history_path.exists():
    left.image(str(history_path), caption="Training and validation curves")
if matrix_path.exists():
    right.image(str(matrix_path), caption="Confusion matrix")

section_title(
    "Feature signal",
    "Signals most associated with churn",
    "This chart uses dataset-level correlation as an interpretable signal ranking. It complements the neural model by making key business drivers easier to explain.",
)
st.plotly_chart(fig_feature_signal(df), use_container_width=True)

report_path = ROOT_DIR / "models" / "classification_report.txt"
if report_path.exists():
    section_title("Classification report", "Detailed test-set metrics")
    st.code(report_path.read_text(encoding="utf-8"), language="text")

footer()
