# Run FinGuard AI on this computer

Project folder: `D:\Bank-Churn-Rate-Predictor`

## Everyday use

1. Open this folder in File Explorer.
2. Double-click **Run_Project.bat**.
3. The app opens at **http://127.0.0.1:8501**. The first page can take a little longer while TensorFlow loads.
4. Keep the launcher window open while using the app. Press **Ctrl+C** in that window to stop, or double-click **Stop_Project.bat**.

After restarting Windows, double-click Run_Project.bat again. Installation and model training are not required each time. This is an on-demand local installation, not an automatically starting Windows service.

If the launcher says the port is in use, first visit the address above; the project may already be running. Stop_Project.bat stops only Python Streamlit processes whose command line identifies this project's exact app.py path. It does not stop other Python projects.

The server listens only on this computer. `127.0.0.1` and `localhost` refer to your own computer, not an internet website. Core predictions and charts use the included files and work without an API key or internet connection after installation. External links and some optional component icon/font resources can still require internet.

## What was installed

- The GitHub repository and its dataset, with a locally trained model using the original training script. The original model and charts are backed up in `.cache/repository-originals/` because its reported metrics could not be reproduced from the bundled network.
- A separate Python 3.12 virtual environment in `.venv` on D:. Your system Python installation is not replaced.
- The repository's packages, with scikit-learn 1.9.0 and Keras 3.14.1 matched to the original model metadata, plus their required dependencies. Use the local lock file for the tested package set.
- Windows launch/stop helpers and a local Streamlit configuration.
- A complete project guide in PROJECT_GUIDE.md and a browser-readable copy in PROJECT_GUIDE.html.
- A validation script and report documenting what actually ran.

The virtual environment uses the existing Python 3.12 installation on C:. Keep that base Python installation installed. Do not use the default Python 3.14 to recreate this environment: this repository pins TensorFlow 2.18.0, and its installed Windows wheel is for Python 3.12.

## Manual start in PowerShell

```powershell
Set-Location D:\Bank-Churn-Rate-Predictor
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

This uses the virtual environment directly, so activation and changes to PowerShell's execution policy are unnecessary. Open http://127.0.0.1:8501. The double-click launcher additionally opens the browser automatically.

## Retrain only when you intend to replace the model

Stop the app first. Keep a copy of `models` and the two training chart PNGs in `assets` if you want to compare old and new results. Then run:

```powershell
Set-Location D:\Bank-Churn-Rate-Predictor
.\.venv\Scripts\python.exe train_model.py
```

The default is up to 45 epochs, with early stopping. It overwrites the saved network, preprocessing files, metrics, test predictions, and training charts. Restart the app afterward to clear its cached model. Neural-network training has randomness; the original script fixes the train/test split seed but does not fix all neural-network random seeds, so metrics can change.

## Check the installation

```powershell
Set-Location D:\Bank-Churn-Rate-Predictor
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe verify_local.py
```

The verification script checks the saved model against stored test predictions, the customer scoring path, all pages, prediction submission, empty analytics filters, and one training step in a temporary in-memory model. It does not overwrite the trained model. Its result is saved in `local_verification.json`.

## Recreate the environment if it is removed

Do not run this while the app is open. With Python 3.12 installed:

```powershell
Set-Location D:\Bank-Churn-Rate-Predictor
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local-lock.txt
```

`requirements-local-lock.txt` records the tested Windows/Python 3.12 package versions. `requirements.txt` now matches the key installed scikit-learn, Keras, and Streamlit versions. If transferring to another operating system, start from the direct requirements and verify compatibility again; the local lock includes Windows-specific TensorFlow packages.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Browser cannot connect | Start Run_Project.bat and wait for the local URL. A closed browser tab does not itself stop the server. |
| Port 8501 is occupied | If it is this app, open the existing page or use Stop_Project.bat and restart. If another application owns it, stop that application yourself or run Streamlit with `--server.port=8502`, then open that port. |
| Python or module not found | Use the `.venv\Scripts\python.exe` commands above, or recreate the environment. |
| Old prediction after retraining | Stop and restart the server so Streamlit reloads model artifacts. |
| Missing model or preprocessing file | Restore the matching `models` files from the repository, or retrain the whole model and preprocessor together. |
| Missing data | Restore `data\bank_customer_churn.csv`; the dashboard and inference reference statistics need it. |
| Contact page says a message is captured | This is a demo response only; the repository does not save messages to a database or send email. |

The example `.env.example` is documentation only: the application does not load a `.env` file automatically. The local configuration is in `.streamlit/config.toml`, and the launcher supplies explicit settings. No credentials are required.
