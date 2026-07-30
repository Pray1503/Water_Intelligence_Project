"""Stage 9 - Preprocess Training Data.

Orchestrates src.training_preprocessing against the train/validation/test
splits produced by Stage 8 (scripts/08_build_training_dataset.py): builds a
PreprocessingConfig from config.yaml, fits the preprocessing pipeline on
the training split only, transforms all three splits, and persists the
transformed data, the fitted pipeline, and a preprocessing report.

This script contains no preprocessing logic of its own -- it is a thin
orchestration layer over the public API of src.training_preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap project root to sys.path to enable execution via:
# python scripts/09_preprocess_training_data.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yaml

from src.training_preprocessing import (
    PreprocessingConfig,
    TrainingPreprocessingError,
    preprocess_datasets,
    save_preprocessing_pipeline,
)

logger = logging.getLogger(__name__)

# Identifies the shape/version of this preprocessing stage's output
# contract, persisted in the report so downstream stages and the model
# manifest can record which preprocessing version produced a given
# artifact.
PREPROCESSING_VERSION = "1.0.0"


class PreprocessTrainingDataError(Exception):
    """Raised when this orchestration script cannot load, save, or
    otherwise complete the preprocessing stage safely. Failures inside
    the preprocessing module itself surface as TrainingPreprocessingError
    and are not duplicated here."""


def _project_root() -> Path:
    """Resolve the project root from this script's location."""
    return Path(__file__).resolve().parents[1]


def load_config(project_root: Path) -> dict[str, Any]:
    """Load config/config.yaml."""
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        raise PreprocessTrainingDataError(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise PreprocessTrainingDataError(
            f"Failed to parse {config_path}: {exc}"
        ) from exc
    return config


def load_split(path: Path, name: str) -> pd.DataFrame:
    """Load one split (train/validation/test) written as parquet."""
    if not path.exists():
        raise PreprocessTrainingDataError(f"'{name}' split not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise PreprocessTrainingDataError(f"'{name}' split at {path} is empty")
    logger.info(
        "preprocess_training_data.split_loaded",
        extra={
            "split": name,
            "path": str(path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
    )
    return df


def build_preprocessing_config(config: dict[str, Any]) -> PreprocessingConfig:
    """Construct a PreprocessingConfig from the project configuration."""
    target_engineering_config = config.get("target_engineering", {})
    preprocessing_config_raw = config.get("preprocessing", {})

    try:
        return PreprocessingConfig(
            target_column=target_engineering_config.get("target_column_name", "target"),
            group_column=target_engineering_config.get(
                "group_column", "District LGD Code"
            ),
            date_column=target_engineering_config.get("date_column", "Date"),
            numeric_impute_strategy=preprocessing_config_raw.get(
                "numeric_impute_strategy", "median"
            ),
            categorical_impute_strategy=preprocessing_config_raw.get(
                "categorical_impute_strategy", "most_frequent"
            ),
            scale_numeric=bool(preprocessing_config_raw.get("scale_numeric", True)),
            onehot_min_frequency=float(
                preprocessing_config_raw.get("onehot_min_frequency", 0.01)
            ),
            random_state=int(preprocessing_config_raw.get("random_state", 42)),
        )
    except TrainingPreprocessingError as exc:
        raise PreprocessTrainingDataError(
            f"Invalid preprocessing configuration: {exc}"
        ) from exc


def save_transformed_split(df: pd.DataFrame, path: Path, name: str) -> None:
    """Write a transformed split to parquet and verify artifact existence and size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001 - surfaced as a domain error below
        raise PreprocessTrainingDataError(
            f"Failed to write '{name}' split to {path}: {exc}"
        ) from exc

    # Sanity check file existence and non-zero size after write
    if not path.exists() or path.stat().st_size == 0:
        raise PreprocessTrainingDataError(
            f"File verification failed: '{name}' split not found or empty at {path}"
        )

    logger.info(
        "preprocess_training_data.split_saved",
        extra={"split": name, "path": str(path), "rows": int(len(df))},
    )


def build_preprocessing_report(
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    metadata_dict: dict[str, Any],
    output_paths: dict[str, str],
    execution_time_seconds: float,
) -> dict[str, Any]:
    """Assemble the preprocessing report contents."""
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_time_seconds": round(execution_time_seconds, 3),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "numeric_feature_count": len(metadata_dict["numeric_columns"]),
        "categorical_feature_count": len(metadata_dict["categorical_columns"]),
        "output_feature_count": len(metadata_dict["feature_names_out"]),
        "pipeline_sha256": metadata_dict.get("pipeline_sha256"),
        "dropped_unsupported_columns": metadata_dict["dropped_unsupported_columns"],
        "excluded_columns": metadata_dict["excluded_columns"],
        "numeric_impute_strategy": metadata_dict["numeric_impute_strategy"],
        "categorical_impute_strategy": metadata_dict["categorical_impute_strategy"],
        "scale_numeric": metadata_dict["scale_numeric"],
        "random_state": metadata_dict["random_state"],
        "output_paths": output_paths,
    }


def save_preprocessing_report(report: dict[str, Any], path: Path) -> None:
    """Persist the preprocessing report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError as exc:
        raise PreprocessTrainingDataError(
            f"Failed to write preprocessing report to {path}: {exc}"
        ) from exc

    if not path.exists() or path.stat().st_size == 0:
        raise PreprocessTrainingDataError(
            f"File verification failed: report not found or empty at {path}"
        )

    logger.info("preprocess_training_data.report_saved", extra={"path": str(path)})


def run_preprocessing_stage(project_root: Path) -> dict[str, Any]:
    """Run the full Stage 9 preprocess-training-data stage end to end."""
    start_time = time.perf_counter()

    logger.info(f"Starting Stage 9 - Training Preprocessing (v{PREPROCESSING_VERSION})")
    logger.info(f"Project root: {project_root}")

    config = load_config(project_root)
    paths_config = config.get("paths", {})

    splits_dir = project_root / paths_config.get("data_splits", "data/splits")
    preprocessed_dir = project_root / paths_config.get(
        "data_preprocessed", "data/preprocessed"
    )
    models_dir = project_root / paths_config.get("models_dir", "models")
    reports_dir = project_root / paths_config.get("reports_dir", "reports")

    logger.info(f"Splits directory: {splits_dir}")
    logger.info(f"Preprocessed output directory: {preprocessed_dir}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Reports directory: {reports_dir}")

    train_df = load_split(splits_dir / "train.parquet", "train")
    validation_df = load_split(splits_dir / "validation.parquet", "validation")
    test_df = load_split(splits_dir / "test.parquet", "test")

    preprocessing_config = build_preprocessing_config(config)

    logger.info("preprocess_training_data.fitting_pipeline")
    result = preprocess_datasets(train_df, validation_df, test_df, preprocessing_config)

    # Save datasets
    train_path = preprocessed_dir / "train.parquet"
    validation_path = preprocessed_dir / "validation.parquet"
    test_path = preprocessed_dir / "test.parquet"
    save_transformed_split(result.train, train_path, "train")
    save_transformed_split(result.validation, validation_path, "validation")
    save_transformed_split(result.test, test_path, "test")

    # Save Pipeline and retrieve the updated metadata containing the calculated SHA-256 hash
    pipeline_path = models_dir / "preprocessing_pipeline.joblib"
    updated_metadata = save_preprocessing_pipeline(
        result.transformer, result.metadata, pipeline_path
    )

    # Verify saved artifacts exist and are non-empty
    metadata_path = models_dir / "preprocessing_metadata.json"
    if not pipeline_path.exists() or pipeline_path.stat().st_size == 0:
        raise PreprocessTrainingDataError(
            f"Pipeline verification failed: file missing or empty at {pipeline_path}"
        )
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise PreprocessTrainingDataError(
            f"Metadata verification failed: file missing or empty at {metadata_path}"
        )

    # Convert the returned updated metadata (with pipeline_sha256) directly to dict
    metadata_dict = updated_metadata.to_dict()

    output_paths = {
        "train": str(train_path),
        "validation": str(validation_path),
        "test": str(test_path),
        "preprocessing_pipeline": str(pipeline_path),
        "preprocessing_metadata": str(metadata_path),
    }

    elapsed_time = time.perf_counter() - start_time

    report = build_preprocessing_report(
        train_rows=int(len(result.train)),
        validation_rows=int(len(result.validation)),
        test_rows=int(len(result.test)),
        metadata_dict=metadata_dict,
        output_paths=output_paths,
        execution_time_seconds=elapsed_time,
    )

    report_path = reports_dir / "preprocessing_report.json"
    save_preprocessing_report(report, report_path)
    report["report_path"] = str(report_path)

    return report


def main() -> None:
    """Entry point for standalone script execution."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    project_root = _project_root()

    try:
        report = run_preprocessing_stage(project_root)
    except (PreprocessTrainingDataError, TrainingPreprocessingError):
        logger.exception("preprocess_training_data.failed")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STAGE 9: PREPROCESSING SUCCEEDED")
    logger.info(f"Execution time: {report['execution_time_seconds']}s")
    logger.info(f"Train rows: {report['train_rows']}")
    logger.info(f"Validation rows: {report['validation_rows']}")
    logger.info(f"Test rows: {report['test_rows']}")
    logger.info(f"Numeric features: {report['numeric_feature_count']}")
    logger.info(f"Categorical features: {report['categorical_feature_count']}")
    logger.info(f"Total output features: {report['output_feature_count']}")
    if report.get("pipeline_sha256"):
        logger.info(f"Pipeline SHA-256: {report['pipeline_sha256'][:12]}...")

    if report["dropped_unsupported_columns"]:
        logger.warning(
            f"Dropped unsupported columns: {report['dropped_unsupported_columns']}"
        )

    logger.info("-" * 60)
    logger.info("Generated Artifacts:")
    logger.info(f" - Pipeline: {report['output_paths']['preprocessing_pipeline']}")
    logger.info(f" - Report:   {report['report_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
