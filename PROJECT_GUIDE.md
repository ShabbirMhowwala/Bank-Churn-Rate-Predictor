# FinGuard AI: understand your bank churn project

This guide explains the actual code in `D:\Bank-Churn-Rate-Predictor`, cloned from [your GitHub repository](https://github.com/ShabbirMhowwala/Bank-Churn-Rate-Predictor). Read LOCAL_SETUP.md first if you only want to launch it. The repository snapshot is commit `897172692a4f320ba6e108e7ee5399e82e6f4584`; local setup additions and the inference correction are described below.

## 1. What problem does it solve?

**Customer churn** means a customer leaves a service. Here, the recorded target is `Exited`: **1 means exited; 0 means stayed**. The application estimates a new customer's exit risk using their profile and banking attributes. A bank could use such a score to prioritize a human review or retention conversation.

This is **supervised learning** because training examples include the known answer. It is **binary classification** because there are two target classes. It uses a **feed-forward artificial neural network (ANN)**, a type of deep learning model.

The dashboard's historical churn rate and an individual prediction are different quantities. Historical churn rate is the fraction of dataset customers who actually exited. The prediction is the neural network's estimated score for one profile. There is no time horizon such as “will leave within 30 days” in the code or dataset schema, so do not claim one.

The project is a local educational prototype. It does not connect to a bank, stream live transactions, execute retention actions, or send customer messages. “Live inference” means a prediction is calculated when you submit the form.

## 2. The whole system at a glance

```text
TRAINING (only when creating or replacing the model)

Historical CSV with Exited labels
    -> remove identifiers and engineer five features
    -> split 80% development / 20% test
    -> fit scaler and category encoder on the development portion
    -> reserve 15% of that portion for validation during model.fit
    -> train ANN using class weights and early stopping
    -> evaluate on the untouched test labels
    -> save network + preprocessor + metrics + plots

PREDICTION (each form submission)

10 customer inputs in browser
    -> Streamlit Python code
    -> same five engineered features, using the reference median
    -> saved scaler and category encoder transform the inputs
    -> saved ANN computes a score from 0 to 1
    -> risk band + rule-based suggestions shown in the browser
```

Training learns the weights. Prediction reuses them; submitting the form does not retrain the network or add the customer to the CSV.

## 3. Technologies and exactly where each is used

| Technology | Job in this project |
|---|---|
| Python 3.12 | Main programming language; runs training, preprocessing, and the web application. |
| TensorFlow 2.18 / Keras | Define, optimize, save, load, and execute the neural network. Keras is the higher-level modeling interface used through TensorFlow. |
| scikit-learn 1.9.0 | Train/test splitting, numeric standardization, one-hot encoding, class weights, and evaluation metrics. The original requirements requested 1.6.1, but the bundled preprocessor was saved with 1.9.0. |
| Pandas 2.2.3 | Reads CSVs into DataFrames, adds features, filters records, groups customers, and computes chart summaries. |
| NumPy 2.0.2 | Numeric arrays, class calculations, probability arrays, and thresholding predictions. |
| Streamlit | Turns Python scripts into browser pages, forms, sliders, charts, and tables; also manages caching and reruns. No separate Flask/Django backend is used. |
| Plotly 6.7.0 | Interactive bars, scatter plots, histograms, and ROC charts in the current multipage interface. |
| Matplotlib 3.10.0 | Saves training curves as PNG images. |
| Seaborn 0.13.2 | Draws the confusion-matrix heatmap on a Matplotlib figure. |
| Joblib 1.4.2 | Saves and loads the fitted preprocessor, feature-column list, and metric dictionary as `.pkl` files. |
| HDF5 / h5py | Underlying format/library used by the `.h5` saved Keras network. |
| streamlit-option-menu | Custom sidebar navigation. |
| streamlit-lottie | Displays the included animation on the About page. |
| streamlit-extras | Metric-card styling helpers used by the theme. |
| HTML and CSS | Custom cards, colors, gradients, layout, and animations embedded into Streamlit. Most styling is in `assets/css/theme.css`. |
| CSV files | Local data storage. There is no SQL database in this implementation. |
| Git / GitHub | Version control and repository hosting; GitHub is not required for each prediction. |
| Python venv and pip | Keep project packages separate from other Python projects and install their dependencies. |
| Windows batch / PowerShell | Local launch and stop helpers added for this installation. |

Other packages in the local lock file are supporting dependencies, such as SciPy for scientific routines, Pillow for images, and web-server libraries pulled in by Streamlit. `requirements-local-lock.txt` records the exact installed versions, including Streamlit and Keras. The updated direct requirements now pin these versions too.

No OpenAI API, chatbot, generative AI, Node.js project, Docker container, or cloud account is needed. The word “AI” here refers to a trained classification network.

## 4. Understand the data

The CSV is `data/bank_customer_churn.csv`. Each row represents one historical customer. The reference dataset contains 10,000 rows and 14 columns, with 2,037 exited customers (20.37%). France, Germany, and Spain are the recorded country categories. The local verification report checks these counts and missing values directly.

| Column | Meaning and treatment |
|---|---|
| RowNumber | Row identifier; dropped. |
| CustomerId | Customer identifier; dropped. |
| Surname | Customer surname; dropped. |
| CreditScore | Numeric credit score. |
| Geography | Country; encoded categorically. |
| Gender | Recorded gender category; encoded categorically. |
| Age | Age in years. |
| Tenure | Years with the bank. |
| Balance | Account balance. |
| NumOfProducts | Number of banking products used. |
| HasCrCard | Whether the customer has a credit card: 1/0. |
| IsActiveMember | Whether the customer is active: 1/0. |
| EstimatedSalary | Estimated salary. |
| Exited | Target label, excluded from the model inputs. |

Dropping names and identifiers prevents the network from treating arbitrary IDs as meaningful customer behavior. There are **10 original input features** after removing the three identifiers and the target. The repository does not specify the monetary currency or establish the dataset's original collection method; do not label the amounts as rupees or invent a data source. Its README refers to a previous Churn_Modelling.csv project.

The provided data has no missing values according to local validation. The training code has no general imputation pipeline. A new dataset containing missing values would require additional handling.

## 5. Feature engineering: turn raw inputs into useful ratios

`utils/preprocessing.py` adds five columns:

| Feature | Actual calculation |
|---|---|
| BalanceSalaryRatio | Balance / EstimatedSalary; a zero salary is replaced by 1 in this calculation. |
| CreditScoreAgeRatio | CreditScore / Age. |
| BalancePerProduct | Balance / NumOfProducts; a zero product count is replaced by 1. |
| TenureByAge | Tenure / Age, except zero tenure is replaced by 1. This is the script's actual convention. |
| IsHighValueInactive | 1 when balance is above the reference median and IsActiveMember is 0; otherwise 0. |

These are engineered hypotheses, not guaranteed causal explanations. They bring information from combinations of columns into explicit inputs the model can learn from.

After engineering, there are 15 feature columns. Thirteen are numeric; the two categorical columns become five one-hot columns: three countries and two genders. The saved model therefore receives **18 numbers per customer**.

**StandardScaler** changes each numeric value to roughly `(value - training_mean) / training_standard_deviation`. It does not put everything into the range 0 to 1. Scaling keeps very large salary/balance values from dominating optimization merely because of their units.

**OneHotEncoder** represents categories as indicator columns. For example, France/Germany/Spain becomes a three-position vector. `handle_unknown="ignore"` lets an unknown category become all zeros instead of causing an encoder error; it does not mean predictions for unseen countries are reliable. The form restricts choices to the recorded countries.

**ColumnTransformer** combines numeric scaling and categorical encoding into one fitted preprocessing object. At prediction time, use `transform`, never `fit_transform`, because new customers must use the same learned scaling and category mapping as training.

## 6. What the neural network does

`train_model.py` defines this architecture:

```text
18 input values
    -> Dense(64, ReLU)
    -> BatchNormalization
    -> Dropout(25%)
    -> Dense(32, ReLU)
    -> Dropout(20%)
    -> Dense(16, ReLU)
    -> Dense(1, sigmoid)
    -> exit-risk score
```

A dense neuron computes a weighted sum of its inputs plus a bias. The hidden layers learn combinations of customer attributes. **ReLU** returns `max(0, value)`, introducing nonlinearity. Without nonlinear activations, stacked dense layers would collapse into an equivalent linear transformation.

**Batch normalization** normalizes intermediate activations during training and uses learned/stored statistics during inference. **Dropout** randomly disables some activations during training to reduce dependence on particular paths. Dropout is disabled during normal prediction.

The last **sigmoid** maps a single number to the interval 0 to 1. A score of 0.72 is displayed as 72%. This is a model score interpreted as estimated exit probability, not proof that the customer will exit and not a 72% accuracy or confidence guarantee. Class weighting and lack of calibration are additional reasons not to treat the score as a precisely validated real-world probability.

The optimizer is **Adam**, with learning rate **0.001**. The loss is **binary cross-entropy**, which penalizes incorrect probability predictions. Backpropagation calculates how each parameter contributed to loss; the optimizer updates parameters to reduce it.

## 7. Training, validation, and test data

The script first splits data into 80% development data (8,000 customers) and 20% test data (2,000). `stratify=y` approximately preserves churn/non-churn proportions. `random_state=42` makes this split repeatable.

Keras then reserves 15% of the 8,000 development examples for validation: about 1,200 validation and 6,800 examples used for weight updates. Validation helps choose when to stop; the test set estimates performance after model selection.

Defaults are **45 maximum epochs** and **64 examples per batch**. An epoch is one pass through the training examples. A batch is a smaller group used for one optimizer update. Early stopping watches **validation AUC**, stops after eight epochs without improvement, and restores weights from the best observed validation AUC.

Only about one fifth of customers exited, so the classes are imbalanced. `compute_class_weight("balanced", ...)` assigns greater weight to errors on the less frequent churn class. This can improve attention to churn cases while increasing false alarms. It is not SMOTE, oversampling, or a guarantee of better accuracy.

Two evaluation-design limitations remain in the original trainer:

1. The high-value balance median is calculated before the train/test split, so that unsupervised statistic uses test feature information.
2. The scaler/encoder is fitted to all 8,000 development rows before Keras takes its validation subset, so validation preprocessing statistics are not isolated from training.

A stronger future implementation would split raw records into train/validation/test first, fit all learned preprocessing statistics only on the true training split, and reuse them unchanged everywhere. The deployed model follows the original training procedure; these evaluation limitations still apply.

## 8. How to read the actual model results

See the verified-results section at the end for precise values reproduced from the saved model and 2,000 held-out rows. `models/metrics.pkl`, `models/test_predictions.csv`, and `models/classification_report.txt` contain the repository's saved evaluation.

| Metric | Question it answers |
|---|---|
| Accuracy | What fraction of all test customers were classified correctly? |
| Precision for Exited | Of the customers predicted to exit, how many actually exited? |
| Recall for Exited | Of the customers who actually exited, how many did we catch? |
| F1 | Harmonic mean of precision and recall, balancing both. |
| ROC AUC | How well do scores rank churn cases above non-churn cases across possible thresholds? |

The confusion matrix has actual classes on rows and predicted classes on columns:

```text
                       Predicted stayed    Predicted exited
Actually stayed        True negatives      False positives
Actually exited        False negatives     True positives
```

A false positive could cause unnecessary outreach. A false negative could miss an at-risk customer. Which error matters more depends on a business cost analysis that is not implemented here.

The repository originally reported 69.10% accuracy, 36.70% churn precision, 71.50% churn recall, and 0.7732 AUC. Those scores could not be reproduced from its bundled network, even after matching the scikit-learn and Keras versions recorded in its files. The original network produced 38.05% accuracy and 0.7422 AUC in this environment, so those old reported metrics are not claimed as verified. The original artifacts were backed up and the same architecture was trained locally using the repository's training script; use the verified local results at the end of this guide for your demonstration.

An always-“stayed” baseline gets 79.65% test accuracy because most customers stayed, but catches zero churners. Always consider both the accuracy baseline and the ability to catch churn, including the cost of false positives. Do not state that deep learning beats logistic regression or tree models; no such trained-model comparison exists in the repo.

Evaluation uses a **0.50 binary threshold**. The UI uses different display bands:

- **Stable:** score below 0.35.
- **Watchlist:** score at least 0.35 and below 0.65.
- **High Risk:** score at least 0.65.

These bands are coded choices, not thresholds optimized on a business objective. A Watchlist score can be on either side of the evaluation threshold. ROC AUC is not the same thing as accuracy.

## 9. Trace one prediction through the code

1. Open **Predict** and fill ten inputs in `pages/2_Predict.py`.
2. The form submit button builds a Python dictionary named `values`.
3. `score_customer()` in `utils/app_data.py` loads the saved network, preprocessor, and feature list. Streamlit's resource cache reuses these objects across subsequent predictions.
4. `make_prediction_frame()` creates a single-row DataFrame in the required raw input order.
5. `add_fintech_features()` produces the five derived columns. The local inference correction uses the reference dataset's balance median.
6. The saved preprocessor transforms the row into 18 numeric inputs.
7. `model.predict()` runs the network and returns a score.
8. `risk_band()` chooses a display label and a fixed suggested action.
9. `explain_customer()` adds hand-written rules, such as mentioning inactivity or a single product.
10. Streamlit renders the score, progress bar, label, and suggestions in your browser.

The “explanation” text is **rule-based**, not SHAP, not an explanation of the neural network's exact reasoning, and not generated by a language model. The feature-signal chart shows absolute dataset correlations with churn; it is not model feature importance or evidence of causation.

## 10. Understand each page

| Page | What it actually shows |
|---|---|
| Home (`app.py`) | Historical customer count and churn rate, saved AUC/recall, and country/product/balance charts. |
| Dashboard | Historical exited count, saved AUC, average balance, and segmentation charts. |
| Predict | Interactive scoring of one manually entered profile. |
| Analytics | Filters the historical CSV by country and activity status, then recalculates charts and summaries. |
| Model Performance | Saved metrics, ROC computed from saved test predictions, training PNGs, correlations, and classification report. |
| About | Project description, workflow, technology list, and a local animation. |
| Contact | Demo form and portfolio placeholders; there is no email sending or persistent storage. |

The Contact submit handler only displays a success message. Do not describe it as a working contact-storage feature. Some branding/portfolio content still links to `harsh31415926` or uses placeholders; it is content inherited from the repo, so review attribution and personalize it appropriately before presenting.

## 11. File map: what to read and in what order

| File or folder | Purpose |
|---|---|
| `data/bank_customer_churn.csv` | Historical labeled data. |
| `data/sample_customers.csv` | Four sample input profiles for manual demos. |
| `utils/preprocessing.py` | Loading, dropping identifiers, feature engineering, scaling/encoding, prediction row creation. |
| `train_model.py` | Defines ANN, trains, evaluates, writes model and charts. |
| `models/customer_exit_risk_model.h5` | Saved network architecture and learned weights. |
| `models/preprocessor.pkl` | Fitted scaler and encoder; must stay paired with the model. |
| `models/feature_columns.pkl` | Expected engineered input column names and order. |
| `models/metrics.pkl` | Saved accuracy, precision, recall, and AUC. |
| `models/test_predictions.csv` | Actual labels, predicted classes, and scores on the held-out set. |
| `models/classification_report.txt` | Precision, recall, F1, and support by class. |
| `utils/app_data.py` | Cached loaders, inference, risk bands, rule explanations, and chart functions. |
| `app.py` and `pages/` | Current application entry point and six pages. |
| `components/ui.py` | Sidebar, reusable cards, hero, animation, and footer. |
| `components/theme.py` | Page configuration and theme injection. |
| `assets/` | CSS, Lottie JSON, and saved evaluation images. |
| `streamlit_app/dashboard.py` | Older alternative dashboard implementation; it is not the entry point used by Run_Project.bat. |
| `notebooks/` | Placeholder README for future experiments; no completed notebook is included. |
| `requirements.txt` | Direct package requirements aligned with the installed versions. |
| `.streamlit/config.toml` | Added local address, port, privacy setting, and dark base theme. |
| `Run_Project.bat`, `Stop_Project.bat` | Added Windows controls for everyday use. |
| `verify_local.py`, `local_verification.json` | Added executable checks and their measured output. |

Suggested reading order: sample CSV -> preprocessing -> model training -> app_data scoring -> Predict page -> app.py -> remaining pages and styling.

## 12. What was corrected locally and why?

In the original inference code, `add_fintech_features()` was called on a single customer. That customer's balance was also the median of the one-row input. Because a value is never strictly greater than itself, `IsHighValueInactive` was always 0 during single-customer scoring, even for a wealthy inactive customer. During training, the flag could be 1 because the reference median came from many customers.

The local change adds an optional `balance_median` argument to feature engineering and passes the historical CSV's median from both prediction implementations. This matches the feature calculation used by the bundled model and original trainer. The regression check compares the transformed single customer against the same customer's training representation and score.

The dataset and original training architecture/script are unchanged. On the original D: installation, the original model files and evaluation charts are preserved in `.cache/repository-originals/`. This private backup is excluded from Git. A fresh network and its matching metrics/charts were generated locally with `train_model.py` because the original model did not reproduce the original reported test scores. The package versions were also matched to the original artifact metadata: scikit-learn 1.9.0 and Keras 3.14.1. These choices and all resolved dependencies are recorded in the local lock file.

Changing the reference dataset without retraining can still invalidate the feature relationship. A stronger future design would store the training-only median alongside the model artifacts. The repository update includes the tested setup, corrected inference, documentation, and the college presentation.

## 13. A five-minute college demonstration

1. **Problem (30 seconds):** “We want to identify bank customers at risk of leaving, so a retention team can prioritize attention.”
2. **Data (45 seconds):** Show the dataset count and churn rate. Explain the ten input fields and binary target.
3. **Prediction (60 seconds):** Open Predict, enter a profile from sample_customers.csv, submit it, and explain the score and risk band. Then change activity or product count and submit again. Do not promise that every individual change moves the score in a particular direction.
4. **Pipeline (60 seconds):** Explain feature engineering, scaling, encoding, the 64/32/16 ANN, and sigmoid output.
5. **Evaluation (60 seconds):** Open Model Performance. Explain recall, false positives, and the majority-class baseline. Be candid about the model's current limitations.
6. **Engineering (30 seconds):** Show that the saved preprocessor and model are reused and that the app runs locally without external inference APIs.
7. **Next work (15 seconds):** Propose leakage-free validation, simpler baselines, threshold tuning, probability calibration, and model-specific explanation methods.

## 14. Viva questions with answers you can defend

**Why is this classification rather than regression?** The target is a discrete exited/stayed label, although the network first produces a continuous probability-like score.

**Why use sigmoid in the final layer?** There is one binary outcome, and sigmoid maps the output into the interval 0 to 1.

**Why use binary cross-entropy?** It measures error between binary labels and predicted probabilities and supports gradient-based training.

**Why scale numeric inputs?** Their units and magnitudes differ sharply; standardization generally makes gradient-based optimization easier.

**Why one-hot encode country and gender?** The dense network needs numeric input, and one-hot vectors avoid inventing a numeric order among categories.

**Why save the preprocessor?** The model expects exactly the transformations used in training. Relearning a scaler or changing category order at inference would change the meaning of the inputs.

**How do you reduce overfitting?** The architecture uses dropout, batch normalization, and early stopping. Whether these are sufficient must be judged on properly isolated validation/test data.

**Why is accuracy insufficient?** An always-stayed classifier already achieves almost 80% here. Recall and precision reveal whether churners are caught and how many alerts are false.

**Is ANN the best choice?** That has not been established. Logistic regression, random forest, and gradient-boosted trees should be compared under the same clean splits and evaluation criteria.

**Does a 70% score mean 70% accuracy?** No. It is one customer's model output. Accuracy is a dataset-wide fraction of correct classifications, and probability calibration has not been established here.

**Does the app learn from every form entry?** No. Training happens only when running the training script. The form performs inference with saved weights.

**Are the suggested actions personalized by another AI model?** No. They are fixed text selected by thresholds and hand-written conditions.

**Does it use a database or API?** It uses CSV files and local artifacts. There is no application database, external prediction API, or live bank integration.

**What is deployment here?** Running a local Streamlit server that loads the saved model and serves the browser interface on this computer.

**What needs improvement before real use?** Proper data provenance, leakage-free evaluation, fair and representative data, calibration and threshold selection, baseline comparison, drift monitoring, robust validation, and measured business outcomes.

## 15. A practical study sequence

1. Learn the meaning of every CSV column and calculate the churn rate yourself.
2. Trace one sample customer's original and engineered features.
3. Understand `fit_transform` versus `transform` and why preprocessing is saved.
4. Draw the ANN and explain ReLU, sigmoid, loss, optimizer, batch, and epoch in your own words.
5. Read the confusion matrix and explain the accuracy/recall tradeoff.
6. Follow a form submission through the code to the displayed score.
7. Practice the demonstration and viva questions, including honest limitations.

You have understood the project when you can explain why the same customer must be transformed identically during training and prediction, and why a good-looking dashboard is separate from evidence that its model works well.

## 16. Verified local results — 23 October 2026

These results come from the locally trained network, saved to disk and then loaded again for verification. The original training script ran for 32 epochs before early stopping. The architecture and training defaults were not changed.

| Measure | Verified result |
|---|---|
| Historical dataset | 10,000 customers; 2,037 exited; 20.37% churn |
| Missing values in supplied CSV | 0 |
| Test set | 2,000 customers: 1,593 stayed and 407 exited |
| Accuracy | 77.40% |
| Churn precision | 46.61% |
| Churn recall | 75.92% |
| ROC AUC | 0.85898 |
| Always-stayed baseline accuracy | 79.65%, with 0% churn recall |
| True negatives | 1,239 stayed customers correctly classified |
| False positives | 354 stayed customers incorrectly flagged as exited |
| False negatives | 98 exited customers missed |
| True positives | 309 exited customers caught |
| Reference balance median | 97,198.54 in the dataset's unspecified monetary units |
| Reopened model vs saved test probabilities | Maximum difference about 0.00000014, within numeric tolerance |

**What you should say:** “On this held-out split, the model catches about 76% of actual churn cases, but only about 47% of its churn alerts are correct. Its accuracy is below an always-stayed baseline, while that baseline catches no churn cases. AUC measures its ranking ability. We still need cleaner validation, threshold tuning, calibration, and comparisons with simpler classifiers.”

### Try the supplied sample profiles

Use the complete input values in `data/sample_customers.csv`. Row numbers below refer to data rows, excluding the header.

| Sample | Recognizable inputs | Local score | Display band |
|---|---|---|---|
| 1 | France; female; age 42; score 619; balance 0; 1 product; active | 65.87% | High Risk |
| 2 | France; female; age 42; score 502; balance 159,660.80; 3 products; inactive | 98.72% | High Risk |
| 3 | Germany; male; age 36; score 850; balance 128,000; 2 products; active | 7.57% | Stable |
| 4 | Spain; female; age 51; score 580; balance 99,000; 1 product; inactive | 84.58% | High Risk |

These scores belong to this saved model. Retraining may change them; they are not known ground-truth outcomes for the sample profiles.

### Verification completed

- Saved model predictions and metrics agree with the training script's saved test output.
- The inactive/high-balance single-customer feature matches its training representation.
- Home and all six named pages execute without app exceptions.
- The prediction form submits and renders an exit probability.
- Empty country filters on Analytics do not crash the page.
- A separate in-memory network performs a training step without changing the deployed model.
- The package manager reports no broken dependency requirements.
- Browser navigation and a real prediction submission work; the default profile displayed 29.4% (Stable). Screenshots are saved in `assets/local_dashboard.png` and `assets/local_prediction.png`.
- Stop_Project.bat's stop helper and Run_Project.bat were exercised successfully; the server was restarted and its local health endpoint returned `ok`.

The machine-readable details are in `local_verification.json`. The training log is `local_training.log`, and the app-check log is `local_validation.log`.
