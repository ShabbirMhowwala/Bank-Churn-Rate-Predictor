"""Modern Streamlit dashboard for bank customer exit-risk predictions."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

from utils.preprocessing import add_fintech_features, make_prediction_frame


DATA_PATH = ROOT_DIR / "data" / "bank_customer_churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "customer_exit_risk_model.h5"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = ROOT_DIR / "models" / "feature_columns.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.pkl"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fb;
        }
        [data-testid="stAppViewContainer"] {
            background: #f7f9fb;
        }
        .block-container {
            padding-top: 7rem;
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] span {
            color: #111827;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: #0f172a;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
            color: #334155;
        }
        [data-testid="stButton"] button,
        [data-testid="stButton"] button * {
            color: #ffffff !important;
        }
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFormSubmitButton"] button * {
            color: #ffffff !important;
            background: #111827;
            border-color: #ef4444;
        }
        [data-testid="stSidebar"] {
            background: #0e1726;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        .metric-card {
            padding: 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
        }
        .risk-low {
            color: #047857;
            font-weight: 700;
        }
        .risk-medium {
            color: #b45309;
            font-weight: 700;
        }
        .risk-high {
            color: #b91c1c;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return None, None, None
    model = keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, preprocessor, feature_columns


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def probability_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High exit risk", "risk-high"
    if probability >= 0.35:
        return "Medium exit risk", "risk-medium"
    return "Low exit risk", "risk-low"


def explain_result(values: dict, probability: float) -> list[str]:
    reasons = []
    if values["IsActiveMember"] == 0:
        reasons.append("Inactive customers usually need faster retention outreach.")
    if values["NumOfProducts"] == 1:
        reasons.append("Single-product customers may have weaker platform stickiness.")
    if values["Balance"] > 100000 and values["EstimatedSalary"] < 100000:
        reasons.append("High balance compared with salary can indicate higher switching sensitivity.")
    if values["CreditScore"] < 600:
        reasons.append("Lower credit score can increase financial vulnerability signals.")
    if probability < 0.35:
        reasons.append("The profile looks comparatively stable against the training data.")
    return reasons[:3]


def render_overview(df: pd.DataFrame) -> None:
    st.title("FinGuard AI")
    st.caption("Deep learning customer exit-risk intelligence for digital banking teams")

    exited_rate = df["Exited"].mean()
    avg_credit_score = df["CreditScore"].mean()
    avg_balance = df["Balance"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers analyzed", f"{len(df):,}")
    col2.metric("Exit rate", f"{exited_rate:.1%}")
    col3.metric("Avg credit score", f"{avg_credit_score:.0f}")

    st.subheader("Portfolio Story")
    st.write(
        "This project uses a neural network to estimate whether a banking customer is likely "
        "to exit. In a real FinTech workflow, this can help prioritize retention offers, "
        "relationship-manager calls, and financial wellness nudges."
    )

    chart_data = (
        df.groupby("Geography", as_index=False)
        .agg(exit_rate=("Exited", "mean"), avg_balance=("Balance", "mean"))
        .sort_values("exit_rate", ascending=False)
    )
    st.bar_chart(chart_data.set_index("Geography")["exit_rate"])
    st.info(f"Average customer balance in the dataset: {avg_balance:,.2f}")


def render_predictor() -> None:
    st.title("Customer Risk Scoring")
    model, preprocessor, feature_columns = load_artifacts()

    if model is None:
        st.warning(
            "Model artifacts are not available yet. Train locally with "
            "`python train_model.py`, then return to this dashboard."
        )
        return

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        credit_score = col1.slider("Credit score", 300, 900, 650)
        geography = col2.selectbox("Country", ["France", "Germany", "Spain"])
        gender = col3.selectbox("Gender", ["Female", "Male"])

        col4, col5, col6 = st.columns(3)
        age = col4.slider("Age", 18, 92, 38)
        tenure = col5.slider("Tenure", 0, 10, 4)
        balance = col6.number_input("Balance", min_value=0.0, value=85000.0, step=1000.0)

        col7, col8, col9 = st.columns(3)
        products = col7.selectbox("Products", [1, 2, 3, 4])
        has_card = col8.toggle("Has credit card", value=True)
        active = col9.toggle("Active member", value=True)

        salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0)
        submitted = st.form_submit_button("Score customer")

    if submitted:
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

        raw_frame = make_prediction_frame(values, feature_columns[:10])
        prediction_frame = add_fintech_features(raw_frame)
        prepared = preprocessor.transform(prediction_frame)
        probability = float(model.predict(prepared, verbose=0).ravel()[0])
        label, css_class = probability_label(probability)

        st.subheader("Prediction Result")
        c1, c2 = st.columns([1, 2])
        c1.metric("Exit probability", f"{probability:.1%}")
        c2.markdown(f"<p class='{css_class}'>{label}</p>", unsafe_allow_html=True)
        st.progress(min(max(probability, 0.0), 1.0))

        st.write("Key explanation signals")
        for reason in explain_result(values, probability):
            st.write(f"- {reason}")


def render_model_room() -> None:
    st.title("Model Room")

    if METRICS_PATH.exists():
        metrics = joblib.load(METRICS_PATH)
        cols = st.columns(4)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_", " ").title(), f"{value:.3f}")
    else:
        st.info("Train the model to populate live evaluation metrics.")

    history_path = ROOT_DIR / "assets" / "training_history.png"
    matrix_path = ROOT_DIR / "assets" / "confusion_matrix.png"

    col1, col2 = st.columns(2)
    if history_path.exists():
        col1.image(str(history_path), caption="Training accuracy and loss")
    if matrix_path.exists():
        col2.image(str(matrix_path), caption="Confusion matrix")

    report_path = ROOT_DIR / "models" / "classification_report.txt"
    if report_path.exists():
        st.code(report_path.read_text(encoding="utf-8"), language="text")


def render_data_lab(df: pd.DataFrame) -> None:
    st.title("Data Lab")
    engineered = add_fintech_features(df)

    st.dataframe(engineered.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.subheader("Exit by activity")
    activity = df.groupby("IsActiveMember")["Exited"].mean().rename(index={0: "Inactive", 1: "Active"})
    col1.bar_chart(activity)

    col2.subheader("Exit by products")
    products = df.groupby("NumOfProducts")["Exited"].mean()
    col2.bar_chart(products)


def main() -> None:
    st.set_page_config(page_title="FinGuard AI", page_icon="FI", layout="wide")
    apply_theme()

    df = load_dataset()
    st.sidebar.title("FinGuard AI")
    page = st.sidebar.radio("Navigate", ["Overview", "Predictor", "Model Room", "Data Lab"])

    if page == "Overview":
        render_overview(df)
    elif page == "Predictor":
        render_predictor()
    elif page == "Model Room":
        render_model_room()
    else:
        render_data_lab(df)
