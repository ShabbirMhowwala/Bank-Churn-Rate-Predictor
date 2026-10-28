"""Preprocessing helpers for the customer exit-risk neural network."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"


def load_bank_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the bank customer dataset and validate the target column."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {path}")
    return df


def add_fintech_features(df: pd.DataFrame, balance_median: float | None = None) -> pd.DataFrame:
    """Create simple finance-inspired features from the original bank columns."""
    engineered = df.copy()

    tenure = engineered["Tenure"].replace(0, 1)
    salary = engineered["EstimatedSalary"].replace(0, 1)
    products = engineered["NumOfProducts"].replace(0, 1)

    engineered["BalanceSalaryRatio"] = engineered["Balance"] / salary
    engineered["CreditScoreAgeRatio"] = engineered["CreditScore"] / engineered["Age"]
    engineered["BalancePerProduct"] = engineered["Balance"] / products
    engineered["TenureByAge"] = tenure / engineered["Age"]
    reference_balance = engineered["Balance"].median() if balance_median is None else balance_median
    engineered["IsHighValueInactive"] = (
        (engineered["Balance"] > reference_balance)
        & (engineered["IsActiveMember"] == 0)
    ).astype(int)

    return engineered


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and binary target after removing identifiers."""
    clean = add_fintech_features(df).drop(columns=DROP_COLUMNS, errors="ignore")
    y = clean[TARGET_COLUMN].astype(int)
    X = clean.drop(columns=[TARGET_COLUMN])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for numeric and categorical data."""
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def make_prediction_frame(values: dict, expected_columns: Iterable[str]) -> pd.DataFrame:
    """Create a single-row frame with the same feature order used in training."""
    missing = [column for column in expected_columns if column not in values]
    if missing:
        raise ValueError(f"Missing prediction fields: {missing}")

    return pd.DataFrame([{column: values[column] for column in expected_columns}])
