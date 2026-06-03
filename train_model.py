"""Train a beginner-friendly deep neural network for bank customer exit risk."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from utils.preprocessing import build_preprocessor, load_bank_data, split_features_target


DATA_PATH = ROOT_DIR / "data" / "bank_customer_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
ASSET_DIR = ROOT_DIR / "assets"


def build_model(input_dim: int) -> keras.Model:
    """Create a compact ANN suitable for tabular FinTech risk scoring."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def plot_training_history(history: keras.callbacks.History) -> None:
    """Save accuracy/loss charts for the README and Streamlit dashboard."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "training_history.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a confusion matrix for model evaluation."""
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Exited"],
        yticklabels=["Stayed", "Exited"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train(data_path: Path = DATA_PATH, epochs: int = 45, batch_size: int = 64) -> dict:
    """Train the model and persist all artifacts required by the app."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_bank_data(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_map = dict(enumerate(class_weights))

    model = build_model(input_dim=X_train_prepared.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train_prepared,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weight_map,
        verbose=1,
    )

    probabilities = model.predict(X_test_prepared).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    model.save(MODEL_DIR / "customer_exit_risk_model.h5")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")
    joblib.dump(metrics, MODEL_DIR / "metrics.pkl")

    report = classification_report(y_test, predictions, target_names=["Stayed", "Exited"])
    (MODEL_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "exit_probability": probabilities,
        }
    ).to_csv(MODEL_DIR / "test_predictions.csv", index=False)

    plot_training_history(history)
    plot_confusion_matrix(y_test, predictions)

    print("\nModel evaluation")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report")
    print(report)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FinTech customer exit-risk ANN")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size)
