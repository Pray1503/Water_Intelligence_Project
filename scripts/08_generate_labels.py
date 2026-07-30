"""
Stage 8 Driver: Generate Water Stress Labels & Targets.
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

from src.label_generation import generate_water_stress_labels

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
    logger.info("=== Starting Stage 8: Water Stress Label Generation ===")

    config = load_config()
    paths = config["paths"]

    # Resolution of paths
    features_dir = PROJECT_ROOT / "data" / "features"
    input_path = features_dir / "feature_dataset.parquet"
    output_parquet = features_dir / "feature_dataset_with_labels.parquet"
    output_csv = features_dir / "feature_dataset_with_labels.csv"
    
    report_dir = PROJECT_ROOT / "reports" / "feature_engineering"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "label_summary.json"

    if not input_path.exists():
        logger.error("Feature dataset not found at: %s. Please run Stage 7 first.", input_path)
        sys.exit(1)

    logger.info("Loading feature dataset from: %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded shape: %s", df.shape)

    # Generate labels & targets
    df_labelled, report = generate_water_stress_labels(df)

    # Save output artifacts
    logger.info("Saving labelled dataset to: %s", output_parquet)
    df_labelled.to_parquet(output_parquet, index=False)

    logger.info("Saving labelled CSV to: %s", output_csv)
    df_labelled.to_csv(output_csv, index=False)

    # Save JSON report
    logger.info("Saving label summary report to: %s", report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    elapsed_time = time.perf_counter() - start_time
    logger.info("=== Stage 8 Completed Successfully in %.4fs! ===", elapsed_time)


if __name__ == "__main__":
    main()
