from __future__ import annotations

"""
Stage 7 Entry Script: Build Feature Dataset.

Executes Stage 7 feature engineering pipeline on Stage 6 master dataset,
validates output quality, and exports data files and execution reports.

Pipeline Phase:
    Stage 7 - Feature Engineering

Inputs:
    data/processed/master_dataset.parquet or .csv

Outputs:
    data/features/feature_dataset.parquet and .csv

Reports:
    reports/feature_engineering/feature_summary.json
    reports/feature_engineering/feature_statistics.json
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import build_feature_dataset

# Setup Logging Strategy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _get_default_feature_config() -> Dict[str, Any]:
    """
    Construct default feature engineering configuration using Stage 6 master dataset column names.

    Returns:
        Centralized dictionary containing lag, rolling, and trend configurations.
    """
    return {
        "lags": {
            "groundwater_level": [1, 7, 30],
            "rainfall_mm": [1, 7],
            "air_temperature": [1],
            "relative_humidity": [1],
            "river_level": [1],
        },
        "rolling": {
            "rainfall_mm": {"windows": [7, 30], "stats": ["mean", "sum", "std", "max"]},
            "groundwater_level": {"windows": [7, 30], "stats": ["mean", "std"]},
            "relative_humidity": {"windows": [15], "stats": ["mean"]},
            "air_temperature": {"windows": [30], "stats": ["mean"]},
            "river_level": {"windows": [7], "stats": ["mean"]},
        },
        "trends": {
            "target_columns": [
                "groundwater_level",
                "rainfall_mm",
                "air_temperature",
                "relative_humidity",
                "river_level",
            ],
            "diff_intervals": [1, 7],
            "rainfall_col": "rainfall_mm",
            "temperature_col": "air_temperature",
            "dry_threshold_mm": 1.0,
            "compute_dry_wet_spells": True,
            "compute_rain_temp_ratio": True,
            "compute_anomaly": True,
        },
    }


def main() -> None:
    """
    Main execution workflow for Stage 7 feature engineering pipeline script.

    Raises:
        SystemExit: Exits with code 1 upon unhandled errors or input dataset missing.
    """
    script_start_time = time.perf_counter()
    logger.info("=== Starting Stage 7: Feature Engineering Pipeline Script ===")

    # 1. Path Resolution
    input_parquet = PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet"
    input_csv = PROJECT_ROOT / "data" / "processed" / "master_dataset.csv"

    output_dir = PROJECT_ROOT / "data" / "features"
    report_dir = PROJECT_ROOT / "reports" / "feature_engineering"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    parquet_output = output_dir / "feature_dataset.parquet"
    csv_output = output_dir / "feature_dataset.csv"

    summary_report_path = report_dir / "feature_summary.json"
    stats_report_path = report_dir / "feature_statistics.json"

    # 2. Input Loading & Verification
    if input_parquet.exists():
        logger.info("Loading Stage 6 master dataset from: %s", input_parquet)
        df_master = pd.read_parquet(input_parquet)
    elif input_csv.exists():
        logger.info("Loading Stage 6 master dataset from: %s", input_csv)
        df_master = pd.read_csv(input_csv, parse_dates=["Date"])
    else:
        logger.error(
            "Stage 6 master dataset not found at expected paths:\n"
            " - %s\n - %s\n"
            "Please run Stage 6 pipeline first.",
            input_parquet,
            input_csv,
        )
        sys.exit(1)

    logger.info("Loaded master dataset shape: %s", df_master.shape)
    logger.info("Dataset Columns: %d", len(df_master.columns))

    # 3. Execution Configuration Setup
    config = _get_default_feature_config()

    if not config:
        raise ValueError("Feature configuration cannot be empty.")

    try:
        # 4. Pipeline Execution
        df_features, (feature_count, summary_report) = build_feature_dataset(
            df=df_master,
            feature_config=config,
            group_column="District LGD Code",
            date_column="Date",
            strict=True,
        )

        # 5. Output Artifact Export
        logger.info("Saving Parquet feature dataset to: %s", parquet_output)
        df_features.to_parquet(parquet_output, index=False)

        logger.info("Saving CSV feature dataset to: %s", csv_output)
        df_features.to_csv(csv_output, index=False)

        # 6. JSON Reports Export
        logger.info("Writing summary report to: %s", summary_report_path)
        with open(summary_report_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4)

        logger.info("Generating feature statistics report at: %s", stats_report_path)
        stats_df = df_features.describe(
            include="all",
        )
        stats_dict = stats_df.to_dict()

        stats_json = {
            str(col): {
                str(k): (
                    float(v)
                    if isinstance(v, (int, float)) and pd.notnull(v)
                    else str(v)
                )
                for k, v in metrics.items()
            }
            for col, metrics in stats_dict.items()
        }

        with open(stats_report_path, "w", encoding="utf-8") as f:
            json.dump(stats_json, f, indent=4)

        script_elapsed_time = time.perf_counter() - script_start_time

        logger.info("=" * 60)
        logger.info("FEATURE DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info("Rows                 : %d", len(df_features))
        logger.info("Columns              : %d", len(df_features.columns))
        logger.info("Engineered Features  : %d", feature_count)
        logger.info("Summary Report       : %s", summary_report_path)
        logger.info("Statistics Report    : %s", stats_report_path)
        logger.info("=" * 60)

        logger.info(
            "=== Stage 7 Feature Engineering Completed Successfully in %.4fs! ===",
            script_elapsed_time,
        )

        return

    except Exception:
        logger.exception("Stage 7 Feature Engineering execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
