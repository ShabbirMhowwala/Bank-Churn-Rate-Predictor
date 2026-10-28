"""Verify the bundled model, prediction flow, and all Streamlit pages locally."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest
from tensorflow import keras
from utils.preprocessing import add_fintech_features, split_features_target
from utils.app_data import score_customer, risk_band


def main():
    df = pd.read_csv(ROOT / "data/bank_customer_churn.csv")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = joblib.load(ROOT / "models/preprocessor.pkl")
    model = keras.models.load_model(ROOT / "models/customer_exit_risk_model.h5", compile=False)
    prepared = preprocessor.transform(X_test)
    scores = model(prepared, training=False).numpy().ravel()
    predicted = (scores >= 0.5).astype(int)
    saved = pd.read_csv(ROOT / "models/test_predictions.csv")
    stored_metrics = joblib.load(ROOT / "models/metrics.pkl")
    metrics = dict(accuracy=accuracy_score(y_test, predicted), precision=precision_score(y_test, predicted), recall=recall_score(y_test, predicted), roc_auc=roc_auc_score(y_test, scores))
    report = {
        "rows": len(df), "columns": df.columns.tolist(), "missing_values": int(df.isna().sum().sum()),
        "exited": int(y.sum()), "churn_rate": float(y.mean()),
        "countries": sorted(df.Geography.unique().tolist()),
        "balance_median": float(df.Balance.median()),
        "input_shape": list(model.input_shape), "parameters": model.count_params(),
        "preprocessed_features": preprocessor.get_feature_names_out().tolist(),
        "metrics_saved": stored_metrics, "metrics_recomputed": metrics,
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "majority_baseline_accuracy": float((y_test == 0).mean()),
        "saved_probability_max_difference": float(np.max(np.abs(saved.exit_probability.to_numpy() - scores))),
    }
    assert np.array_equal(saved.actual.to_numpy(), y_test.to_numpy()), "Saved test labels do not match the split."
    np.testing.assert_allclose(saved.exit_probability.to_numpy(), scores, atol=1e-5, rtol=1e-5)
    for key in metrics:
        assert abs(metrics[key] - stored_metrics[key]) < 1e-5, (key, metrics[key], stored_metrics[key])
    samples = pd.read_csv(ROOT / "data/sample_customers.csv")
    report["sample_predictions"] = []
    for values in samples.to_dict("records"):
        score = score_customer(values)
        assert np.isfinite(score) and 0 <= score <= 1
        report["sample_predictions"].append(dict(input=values, probability=score, band=risk_band(score)[0]))

    # Compare the training feature with the single-record feature computation.
    raw_columns = joblib.load(ROOT / "models/feature_columns.pkl")[:10]
    example_index = df[(df.Balance > df.Balance.median()) & (df.IsActiveMember == 0)].index[0]
    one = df.loc[[example_index], raw_columns]
    report["single_row_feature_check"] = {
        "training_high_value_inactive": int(X.loc[example_index, "IsHighValueInactive"]),
        "default_single_row_high_value_inactive": int(add_fintech_features(one).IsHighValueInactive.iloc[0]),
        "corrected_single_row_high_value_inactive": int(add_fintech_features(one, balance_median=float(df.Balance.median())).IsHighValueInactive.iloc[0]),
    }
    corrected = add_fintech_features(one, balance_median=float(df.Balance.median()))
    np.testing.assert_allclose(preprocessor.transform(corrected), preprocessor.transform(X.loc[[example_index]]))
    expected_score = float(model(preprocessor.transform(X.loc[[example_index]]), training=False).numpy().ravel()[0])
    assert abs(score_customer(one.iloc[0].to_dict()) - expected_score) < 1e-6
    assert report["single_row_feature_check"]["corrected_single_row_high_value_inactive"] == 1

    report["pages"] = {}
    for page in [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]:
        app = AppTest.from_file(str(page), default_timeout=60).run()
        errors = [str(e.message) for e in app.exception]
        report["pages"][str(page.relative_to(ROOT))] = errors or "PASS"
        assert not errors, (page.name, errors)
        if page.name == "2_Predict.py":
            app.button[0].click().run(timeout=60)
            assert not app.exception, [str(e.message) for e in app.exception]
            assert any("Exit probability" in str(item.value) for item in app.markdown)
            report["prediction_form"] = "PASS"
        if page.name == "3_Analytics.py":
            app.multiselect[0].set_value([]).run(timeout=60)
            assert not app.exception, [str(e.message) for e in app.exception]
            report["empty_analytics_filter"] = "PASS"
    # A fresh, unsaved model can perform a training step with this environment.
    from train_model import build_model
    train_model = build_model(prepared.shape[1])
    step = train_model.train_on_batch(prepared[:64], y_test.to_numpy()[:64])
    assert np.isfinite(np.asarray(step)).all()
    report["training_step"] = "PASS (temporary in-memory model; saved artifacts unchanged)"
    (ROOT / "local_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
