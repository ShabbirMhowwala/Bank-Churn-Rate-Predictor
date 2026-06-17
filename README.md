# : CHAT W Deep Learning Customer Exit Risk Predictor

FinGuard AI is a beginner-friendly but portfolio-ready FinTech project that predicts whether a bank customer is likely to leave using a TensorFlow/Keras neural network.

The project is based on the existing `2_Project_Churn/Churn_Modelling.csv` dataset from this Deep Learning folder and upgrades the notebook-style churn concept into a startup-style AI product with training scripts, saved model artifacts, and a Streamlit dashboard.

## Problem Statement

Digital banks and FinTech platforms need to identify customers who are likely to exit before they actually leave. This project scores customer exit risk from profile and banking behavior signals such as credit score, tenure, balance, product count, activity status, and estimated salary.

## FinTech Relevance

- Customer retention analytics for digital banking teams
- Early warning system for high-value inactive customers
- Risk-prioritized outreach for relationship managers
- Practical tabular deep learning workflow for financial customer data

## Deep Learning Approach

The model is an artificial neural network for binary classification:

- Input: preprocessed customer banking features
- Hidden layers: dense layers with ReLU activations
- Regularization: dropout and batch normalization
- Output: sigmoid probability of customer exit
- Loss: binary cross entropy
- Metrics: accuracy, precision, recall, ROC AUC, confusion matrix

## Project Structure

```text
fintech-ai-project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── bank_customer_churn.csv
│   └── sample_customers.csv
├── models/
├── notebooks/
├── utils/
│   └── preprocessing.py
├── assets/
└── streamlit_app/
    └── dashboard.py
```

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## How To Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the neural network:

```bash
python train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Saved Artifacts

Training creates:

- `models/customer_exit_risk_model.h5`
- `models/preprocessor.pkl`
- `models/feature_columns.pkl`
- `models/metrics.pkl`
- `models/classification_report.txt`
- `assets/training_history.png`
- `assets/confusion_matrix.png`

## Sample Inputs

See `data/sample_customers.csv` for example customer profiles. The Streamlit app also includes an interactive form for manual scoring.

## Screenshots

Add screenshots after training and running the app:

- `assets/dashboard_overview.png`
- `assets/prediction_result.png`
- `assets/model_room.png`

## Resume-Worthy Bullets

- Built a TensorFlow/Keras neural network to predict customer exit risk for a banking dataset.
- Engineered finance-specific features such as balance-to-salary ratio and high-value inactive customer flags.
- Created a Streamlit analytics dashboard with real-time prediction, confidence score, explanations, and model evaluation views.
- Saved reusable model and preprocessing artifacts for local deployment and Streamlit Cloud readiness.

## Future Improvements

- Add SHAP-based feature explanations
- Compare ANN against logistic regression, random forest, and XGBoost baselines
- Add threshold tuning for business-friendly recall targets
- Connect the dashboard to a CRM-style customer list
- Add monitoring for model drift and changing exit rates

## Disclaimer

This is an educational portfolio project. It should not be used for real customer decisions without stronger validation, bias checks, monitoring, and compliance review.
