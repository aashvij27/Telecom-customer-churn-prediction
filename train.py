from __future__ import annotations

import argparse
import json

import joblib

from src.config import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, FIGURE_DIR, MODEL_DIR, REPORT_DIR
from src.data import clean_telco_data, load_telco_data, split_features_target
from src.modeling import train_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the telecom churn prediction model.")
    parser.add_argument("--data", default=None, help="Path to the Telco churn CSV file.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the public sample dataset if data/ does not contain the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_telco_data(args.data, download=args.download)
    cleaned_df = clean_telco_data(raw_df)
    X, y = split_features_target(cleaned_df)

    result = train_best_model(X, y)
    joblib.dump(result.pipeline, DEFAULT_MODEL_PATH)

    report = {
        **result.metrics,
        "training_rows": int(len(cleaned_df)),
        "positive_churn_rate": round(float(y.mean()), 4),
        "top_features": result.feature_importance.to_dict(orient="records"),
    }
    DEFAULT_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Best model: {result.model_name}")
    print(json.dumps(result.metrics, indent=2))
    print(f"Saved model to {DEFAULT_MODEL_PATH}")
    print(f"Saved metrics to {DEFAULT_METRICS_PATH}")


if __name__ == "__main__":
    main()
