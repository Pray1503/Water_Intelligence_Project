"""
Stage 9 Driver: Train Water Stress Prediction Models.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import split_chronologically, train_lead_model, save_model_artifact, prepare_training_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    start_time = time.perf_counter()
    logger.info("=== Starting Stage 9: Water Stress Model Training ===")

    config = load_config()
    paths = config["paths"]

    # Paths resolution
    features_dir = PROJECT_ROOT / "data" / "features"
    input_path = features_dir / "feature_dataset_with_labels.parquet"
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    report_dir = PROJECT_ROOT / "reports" / "model_training"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    importances_path = report_dir / "feature_importances.json"

    if not input_path.exists():
        logger.error("Labelled dataset not found at: %s. Please run Stage 8 first.", input_path)
        sys.exit(1)

    logger.info("Loading dataset from: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Dataset shape: %s", df.shape)

    # 1. Split dataset chronologically
    # 2021-2024 is train set, 2025 is validation/test set
    logger.info("Splitting dataset chronologically at 2025-01-01...")
    train_df, test_df = split_chronologically(df, split_date="2025-01-01", date_col="Date")
    logger.info("Train set rows: %d | Test set rows: %d", len(train_df), len(test_df))

    targets = {
        "wsi_lead_7": models_dir / "model_7d.joblib",
        "wsi_lead_15": models_dir / "model_15d.joblib",
        "wsi_lead_30": models_dir / "model_30d.joblib"
    }

    all_metrics = {}
    feature_importances = {}

    # Define features to exclude (targets, identifiers)
    feature_columns_to_exclude = [
        "Date",
        "District",
        "wsi_lead_7",
        "wsi_lead_15",
        "wsi_lead_30"
    ]

    for target_col, model_path in targets.items():
        logger.info("-" * 50)
        logger.info("Training predictor for target: %s", target_col)
        
        # Train model
        model, metrics = train_lead_model(
            train_df,
            test_df,
            target_col,
            feature_columns_to_exclude
        )
        
        # Save model binary
        save_model_artifact(model, model_path)
        all_metrics[target_col] = metrics

        # Extract feature importances
        # Get feature names after preparation
        X_sample, _ = prepare_training_data(train_df.head(10), target_col, feature_columns_to_exclude)
        feature_names = list(X_sample.columns)
        importances = model.feature_importances_
        
        # Sort importances
        sorted_indices = importances.argsort()[::-1]
        top_importances = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in sorted_indices[:15]  # Top 15 features
        ]
        feature_importances[target_col] = top_importances

    # 2. Save evaluation metrics JSON
    logger.info("-" * 50)
    logger.info("Saving evaluation metrics to: %s", metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    # 3. Save feature importances JSON
    logger.info("Saving feature importances to: %s", importances_path)
    with open(importances_path, "w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=4)

    elapsed_time = time.perf_counter() - start_time
    logger.info("=== Stage 9 Completed Successfully in %.4fs! ===", elapsed_time)


if __name__ == "__main__":
    main()
