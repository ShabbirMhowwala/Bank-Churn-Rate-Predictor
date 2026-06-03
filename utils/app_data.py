"""Cached data, model, and chart helpers for the Streamlit app."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, roc_curve
from tensorflow import keras

from utils.preprocessing import add_fintech_features, make_prediction_frame


DATA_PATH = ROOT_DIR / "data" / "bank_customer_churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "customer_exit_risk_model.h5"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = ROOT_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.pkl"
PREDICTIONS_PATH = ROOT_DIR / "models" / "test_predictions.csv"


PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#e5e7eb"},
    "margin": {"l": 30, "r": 20, "t": 50, "b": 30},
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return joblib.load(METRICS_PATH)


@st.cache_resource
def load_model_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return None, None, None
    return (
        keras.models.load_model(MODEL_PATH),
        joblib.load(PREPROCESSOR_PATH),
        joblib.load(FEATURE_COLUMNS_PATH),
    )


def score_customer(values: dict) -> float:
    model, preprocessor, feature_columns = load_model_artifacts()
    if model is None:
        raise FileNotFoundError("Model artifacts are missing. Run python train_model.py first.")

    raw_frame = make_prediction_frame(values, feature_columns[:10])
    prepared = preprocessor.transform(add_fintech_features(raw_frame))
    return float(model.predict(prepared, verbose=0).ravel()[0])


def risk_band(probability: float) -> tuple[str, str, str]:
    if probability >= 0.65:
        return "High Risk", "high-risk", "Launch retention action within 24 hours."
    if probability >= 0.35:
        return "Watchlist", "medium-risk", "Send personalized product and activity nudges."
    return "Stable", "low-risk", "Maintain engagement and monitor future activity."


def explain_customer(values: dict, probability: float) -> list[str]:
    reasons: list[str] = []
    if values["IsActiveMember"] == 0:
        reasons.append("Inactive member status is a strong retention signal.")
    if values["NumOfProducts"] == 1:
        reasons.append("Single-product customers have lower platform stickiness.")
    if values["Balance"] > 100000 and values["EstimatedSalary"] < 100000:
        reasons.append("Large balance relative to salary may indicate switching sensitivity.")
    if values["CreditScore"] < 600:
        reasons.append("Lower credit score adds financial vulnerability context.")
    if probability < 0.35:
        reasons.append("Profile is comparatively stable against the training distribution.")
    return reasons[:4]


def churn_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, as_index=False)
        .agg(churn_rate=("Exited", "mean"), customers=("Exited", "size"), avg_balance=("Balance", "mean"))
        .sort_values("churn_rate", ascending=False)
    )


def fig_churn_by_geo(df: pd.DataFrame) -> go.Figure:
    data = churn_by_category(df, "Geography")
    fig = px.bar(
        data,
        x="Geography",
        y="churn_rate",
        color="churn_rate",
        color_continuous_scale=["#22d3ee", "#6366f1", "#a855f7"],
        text=data["churn_rate"].map(lambda x: f"{x:.1%}"),
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False)
    return fig


def fig_balance_salary(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(1200, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="EstimatedSalary",
        y="Balance",
        color=sample["Exited"].map({0: "Retained", 1: "Exited"}),
        size="Age",
        opacity=0.72,
        color_discrete_map={"Retained": "#22d3ee", "Exited": "#f472b6"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**PLOTLY_LAYOUT, legend_title_text="")
    return fig


def fig_product_churn(df: pd.DataFrame) -> go.Figure:
    data = churn_by_category(df, "NumOfProducts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["NumOfProducts"],
        y=data["churn_rate"],
        mode="lines+markers",
        fill="tozeroy",
        line={"color": "#38bdf8", "width": 4},
        marker={"size": 12, "color": "#a855f7"},
    ))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Products", yaxis_title="Churn rate")
    return fig


def fig_age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Age",
        color=df["Exited"].map({0: "Retained", 1: "Exited"}),
        nbins=28,
        color_discrete_map={"Retained": "#22d3ee", "Exited": "#a855f7"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**PLOTLY_LAYOUT, legend_title_text="")
    return fig


def fig_roc_curve() -> go.Figure:
    preds = load_predictions()
    fig = go.Figure()
    if not preds.empty:
        fpr, tpr, _ = roc_curve(preds["actual"], preds["exit_probability"])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": "#22d3ee", "width": 5}, name=f"AUC {roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#64748b"}, name="Baseline"))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def fig_feature_signal(df: pd.DataFrame) -> go.Figure:
    engineered = add_fintech_features(df).drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
    numeric = engineered.select_dtypes(include=[np.number])
    signal = numeric.corr(numeric_only=True)["Exited"].drop("Exited").abs().sort_values(ascending=True).tail(10)
    fig = px.bar(
        signal,
        x=signal.values,
        y=signal.index,
        orientation="h",
        color=signal.values,
        color_continuous_scale=["#06b6d4", "#6366f1", "#c084fc"],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Absolute correlation with churn", yaxis_title="", coloraxis_showscale=False)
    return fig
