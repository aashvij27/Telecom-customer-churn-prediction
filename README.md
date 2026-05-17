# Telecom Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using demographic, account, and service usage features. It started as a Jupyter notebook and has been refactored into a GitHub-ready machine learning project with a reproducible training script and a Streamlit demo app.

## Project Highlights

- Cleans the IBM/Kaggle Telco Customer Churn dataset.
- Trains and compares Logistic Regression, Random Forest, and Gradient Boosting classifiers.
- Selects the best model using ROC-AUC and F1 score.
- Saves the trained pipeline with preprocessing included.
- Provides a Streamlit interface for single-customer and batch CSV predictions.

## Repository Structure

```text
.
├── app.py                              # Streamlit demo app
├── train.py                            # Model training entry point
├── requirements.txt                    # Python dependencies
├── telecom_Customer_Churn.ipynb        # Original exploratory notebook
├── src/
│   ├── config.py
│   ├── data.py
│   └── modeling.py
├── examples/
│   └── sample_customers.csv             # Small batch-scoring example
├── data/                               # Local CSV goes here, ignored by Git
├── models/                             # Trained model artifacts, ignored by Git
├── reports/                            # Metrics and generated reports
└── docs/                               # Existing project documentation
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train The Model

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` folder, then run:

```bash
python train.py
```

Or let the script download the public sample dataset:

```bash
python train.py --download
```

Training saves:

- `models/churn_model.joblib`
- `reports/metrics.json`

## Run The App

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in the terminal. The app trains the model on first run if `models/churn_model.joblib` is missing.

For batch scoring, upload `examples/sample_customers.csv` or a CSV with the same schema as the Telco churn dataset.

## Deploy

The simplest deployment path is Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Create a new app from the GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy.

Because `app.py` can train from the public CSV on first run, no model artifact needs to be committed.

## Dataset

This project uses the public Telco Customer Churn sample dataset commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Notes

The original notebook is kept for exploratory data analysis and project history. The production demo uses the modular Python files in `src/`.
