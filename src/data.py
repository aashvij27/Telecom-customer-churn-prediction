from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_URL, DEFAULT_DATA_PATH, ID_COLUMN, TARGET_COLUMN


def load_telco_data(path: str | Path | None = None, download: bool = False) -> pd.DataFrame:
    """Load the Telco churn dataset from a local CSV or the public sample URL."""
    csv_path = Path(path) if path else DEFAULT_DATA_PATH

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if not download:
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Add the CSV there or run with --download."
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL)
    df.to_csv(csv_path, index=False)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column types and remove rows that cannot be used for supervised training."""
    cleaned = df.copy()

    if ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[ID_COLUMN])

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["PaymentMethod"] = cleaned["PaymentMethod"].str.replace(
        " (automatic)", "", regex=False
    )
    cleaned = cleaned.dropna(subset=["TotalCharges", TARGET_COLUMN])
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin(["Yes", "No"])]

    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    return features, target
