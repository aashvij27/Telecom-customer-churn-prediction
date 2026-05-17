from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DEFAULT_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "churn_model.joblib"
DEFAULT_METRICS_PATH = REPORT_DIR / "metrics.json"

DATA_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/"
    "master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
