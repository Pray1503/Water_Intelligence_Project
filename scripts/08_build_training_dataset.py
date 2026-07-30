"""Stage 8 - Build Training Dataset.

Orchestrates the first half of the frozen Stage 8 pipeline: loads the
frozen feature dataset, computes its SHA-256 checksum for lineage,
generates the forecast target, runs dataset validation (target validation,
leakage audit, temporal continuity checks), and performs a chronological,
grouped time-based split into train/validation/test sets. Downstream stages
(Training Preprocessing onward) consume the outputs written here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.target_engineering import (
    TargetConfig,
    TargetEngineeringError,
    generate_forecast_target,
    save_transform_metadata,
)

logger = logging.getLogger(__name__)

REPORT_VERSION = "1.0.0"
PIPELINE_STAGE = "Stage8_BuildTrainingDataset"
_NON_FEATURE_COLUMNS_DEFAULT = frozenset({"target"})


class TrainingDatasetBuildError(Exception):
    """Raised when the training dataset cannot be built or validated safely."""


@dataclass(frozen=True)
class SplitRatios:
    """Chronological split proportions.

    Attributes:
        train: Proportion of the date range allocated to training.
        validation: Proportion allocated to validation.
        test: Proportion allocated to test.
    """

    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if not (
            0.0 < self.train < 1.0
            and 0.0 < self.validation < 1.0
            and 0.0 < self.test < 1.0
        ):
            raise TrainingDatasetBuildError(
                f"Split ratios must each be between 0 and 1, got "
                f"train={self.train}, validation={self.validation}, test={self.test}"
            )
        if abs(total - 1.0) > 1e-6:
            raise TrainingDatasetBuildError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train}, validation={self.validation}, test={self.test})"
            )


def _project_root() -> Path:
    """Resolve the project root from this script's location."""
    return Path(__file__).resolve().parents[1]


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file for lineage tracking and dataset verification.

    Args:
        file_path: Path to the target file.
        chunk_size: Reading buffer size in bytes.

    Returns:
        Hexadecimal SHA-256 hash string.

    Raises:
        TrainingDatasetBuildError: If the file cannot be read.
    """
    if not file_path.exists():
        raise TrainingDatasetBuildError(f"Cannot hash non-existent file: {file_path}")

    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                sha256.update(chunk)
    except OSError as exc:
        raise TrainingDatasetBuildError(
            f"Failed to read {file_path} for SHA-256 hashing: {exc}"
        ) from exc

    digest = sha256.hexdigest()
    logger.debug(
        "build_training_dataset.hash_computed",
        extra={"path": str(file_path), "sha256": digest},
    )
    return digest


def load_config(project_root: Path) -> dict[str, Any]:
    """Load and validate config/config.yaml schema structure.

    Raises:
        TrainingDatasetBuildError: If the config file is missing, unparseable,
            or missing required top-level configuration blocks.
    """
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        raise TrainingDatasetBuildError(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise TrainingDatasetBuildError(
            f"Failed to parse {config_path}: {exc}"
        ) from exc

    # Structural Validation
    required_sections = {"paths", "target_engineering", "split"}
    missing_sections = required_sections - set(config.keys())
    if missing_sections:
        raise TrainingDatasetBuildError(
            f"Config file at {config_path} missing required section(s): {sorted(missing_sections)}"
        )

    return config


def load_feature_dataset(path: Path) -> pd.DataFrame:
    """Load the frozen feature dataset.

    Raises:
        TrainingDatasetBuildError: If the file does not exist or is empty.
    """
    if not path.exists():
        raise TrainingDatasetBuildError(f"Feature dataset not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise TrainingDatasetBuildError(f"Feature dataset at {path} is empty")
    logger.info(
        "build_training_dataset.feature_dataset_loaded",
        extra={
            "path": str(path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
    )
    return df


def validate_target(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    """Validate the engineered target column.

    Raises:
        TrainingDatasetBuildError: If the target column is missing or
            entirely null.
    """
    if target_column not in df.columns:
        raise TrainingDatasetBuildError(
            f"Target column '{target_column}' not found after target engineering"
        )

    n_total = int(len(df))
    n_null = int(df[target_column].isna().sum())
    n_valid = n_total - n_null

    if n_valid == 0:
        raise TrainingDatasetBuildError(
            f"Target column '{target_column}' is entirely null; check the "
            f"forecast horizon against the available date range"
        )

    valid_fraction = n_valid / n_total
    if valid_fraction < 0.5:
        logger.warning(
            "build_training_dataset.target_mostly_null",
            extra={
                "target_column": target_column,
                "valid_fraction": round(valid_fraction, 4),
                "n_valid": n_valid,
                "n_total": n_total,
            },
        )

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TrainingDatasetBuildError(
            f"Target column '{target_column}' is not numeric (dtype="
            f"{df[target_column].dtype})"
        )

    return {"n_total": n_total, "n_null": n_null, "n_valid": n_valid}


def run_leakage_audit(
    df: pd.DataFrame,
    target_column: str,
    exclude_columns: frozenset[str],
    correlation_threshold: float = 0.999,
) -> list[str]:
    """Run a lightweight leakage smoke test on candidate feature columns.

    Checks that no feature column is an exact duplicate of the target
    (a strong sign the raw future value leaked into the feature set) and
    flags any feature with suspiciously high absolute correlation to the
    target for manual review.

    Args:
        df: Dataframe including the target column.
        target_column: Name of the engineered target column.
        exclude_columns: Non-feature columns to skip (group/date/target
            and any other identifier columns).
        correlation_threshold: Absolute correlation above which a
            feature is flagged for manual leakage review.

    Returns:
        A list of human-readable warning strings for flagged features
        (empty if none found).

    Raises:
        TrainingDatasetBuildError: If a feature column is an exact
            duplicate of the target column.
    """
    warnings: list[str] = []
    valid_rows = df[df[target_column].notna()]
    if valid_rows.empty:
        raise TrainingDatasetBuildError(
            "No rows with a valid target are available to run the leakage audit"
        )

    candidate_columns = [
        c
        for c in df.columns
        if c not in exclude_columns
        and c != target_column
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for column in candidate_columns:
        if valid_rows[column].equals(valid_rows[target_column]):
            raise TrainingDatasetBuildError(
                f"Feature column '{column}' is identical to the target column "
                f"'{target_column}'. This strongly indicates the raw future "
                f"value leaked into the feature set."
            )

        aligned = valid_rows[[column, target_column]].dropna()
        if len(aligned) < 2:
            continue
        correlation = aligned[column].corr(aligned[target_column])
        if pd.notna(correlation) and abs(correlation) >= correlation_threshold:
            message = (
                f"Feature '{column}' has high |correlation|={abs(correlation):.4f} "
                f"with target '{target_column}' (>= {correlation_threshold}). "
                f"Flagged for manual review (may indicate feature leakage or legitimate strong signal)."
            )
            warnings.append(message)
            logger.warning(
                "build_training_dataset.leakage_risk_flagged", extra={"detail": message}
            )

    if not warnings:
        logger.info("build_training_dataset.leakage_audit_clean")

    return warnings


def run_temporal_continuity_check(
    df: pd.DataFrame, group_column: str, date_column: str
) -> dict[str, Any]:
    """Check per-group date coverage and surface unusually large gaps.

    Raises:
        TrainingDatasetBuildError: If any group has fewer than 2 distinct
            dates, since no temporal ordering can be validated for it.
    """
    gap_warning_threshold_days = 30
    per_group_stats: dict[str, dict[str, Any]] = {}

    for group_value, group_df in df.groupby(group_column):
        dates = pd.to_datetime(group_df[date_column]).sort_values().unique()
        if len(dates) < 2:
            raise TrainingDatasetBuildError(
                f"Group '{group_value}' has fewer than 2 distinct dates; "
                f"cannot validate temporal continuity"
            )
        gaps = pd.Series(dates).diff().dt.days.dropna()
        max_gap = int(gaps.max())
        per_group_stats[str(group_value)] = {
            "min_date": str(pd.Timestamp(dates.min()).date()),
            "max_date": str(pd.Timestamp(dates.max()).date()),
            "n_dates": int(len(dates)),
            "max_gap_days": max_gap,
        }
        if max_gap > gap_warning_threshold_days:
            logger.warning(
                "build_training_dataset.large_temporal_gap",
                extra={
                    "group": str(group_value),
                    "max_gap_days": max_gap,
                    "threshold_days": gap_warning_threshold_days,
                },
            )

    logger.info(
        "build_training_dataset.temporal_continuity_checked",
        extra={"n_groups": len(per_group_stats)},
    )
    return per_group_stats


def compute_time_based_split(
    df: pd.DataFrame,
    date_column: str,
    group_column: str,
    target_column: str,
    ratios: SplitRatios,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Chronologically split the dataset using global date cutoffs.

    Cutoff dates are computed once from the full sorted date range and
    applied uniformly to every group. Rows without a valid target are
    dropped before splitting.

    Returns:
        (train_df, validation_df, test_df, split_report)

    Raises:
        TrainingDatasetBuildError: If too few valid dates remain to form
            three non-empty splits.
    """
    valid_df = df[df[target_column].notna()].copy()
    n_dropped_null_target = int(len(df) - len(valid_df))
    if n_dropped_null_target:
        logger.info(
            "build_training_dataset.dropped_rows_without_target",
            extra={"n_dropped": n_dropped_null_target},
        )

    valid_df[date_column] = pd.to_datetime(valid_df[date_column])
    unique_dates = (
        pd.Series(valid_df[date_column].unique()).sort_values().reset_index(drop=True)
    )

    if len(unique_dates) < 3:
        raise TrainingDatasetBuildError(
            f"Only {len(unique_dates)} distinct dates with a valid target are "
            f"available; cannot form three non-empty chronological splits"
        )

    train_cutoff_idx = int(len(unique_dates) * ratios.train)
    validation_cutoff_idx = int(len(unique_dates) * (ratios.train + ratios.validation))
    train_cutoff_idx = max(1, min(train_cutoff_idx, len(unique_dates) - 2))
    validation_cutoff_idx = max(
        train_cutoff_idx + 1, min(validation_cutoff_idx, len(unique_dates) - 1)
    )

    train_end_date = unique_dates.iloc[train_cutoff_idx - 1]
    validation_end_date = unique_dates.iloc[validation_cutoff_idx - 1]

    train_df = valid_df[valid_df[date_column] <= train_end_date]
    validation_df = valid_df[
        (valid_df[date_column] > train_end_date)
        & (valid_df[date_column] <= validation_end_date)
    ]
    test_df = valid_df[valid_df[date_column] > validation_end_date]

    if train_df.empty or validation_df.empty or test_df.empty:
        raise TrainingDatasetBuildError(
            "Chronological split produced at least one empty split "
            f"(train={len(train_df)}, validation={len(validation_df)}, "
            f"test={len(test_df)}); adjust split ratios or check date coverage"
        )

    # Edge Case Warning: Verify per-group train counts
    group_train_counts = train_df.groupby(group_column).size()
    sparse_train_groups = group_train_counts[group_train_counts < 10].to_dict()
    if sparse_train_groups:
        logger.warning(
            "build_training_dataset.sparse_train_groups_detected",
            extra={"sparse_groups": sparse_train_groups},
        )

    split_report = {
        "n_dropped_null_target": n_dropped_null_target,
        "train_end_date": str(train_end_date.date()),
        "validation_end_date": str(validation_end_date.date()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": [
            str(train_df[date_column].min().date()),
            str(train_df[date_column].max().date()),
        ],
        "validation_date_range": [
            str(validation_df[date_column].min().date()),
            str(validation_df[date_column].max().date()),
        ],
        "test_date_range": [
            str(test_df[date_column].min().date()),
            str(test_df[date_column].max().date()),
        ],
    }

    logger.info("build_training_dataset.split_completed", extra=split_report)
    return train_df, validation_df, test_df, split_report


def build_training_dataset(project_root: Path) -> dict[str, Any]:
    """Run the full build-training-dataset stage end to end."""
    config = load_config(project_root)
    paths_config = config["paths"]
    target_config_raw = config["target_engineering"]
    split_config_raw = config["split"]

    feature_dataset_path = project_root / paths_config.get(
        "feature_dataset", "data/processed/feature_dataset.parquet"
    )
    splits_dir = project_root / paths_config.get("data_splits", "data/splits")
    reports_dir = project_root / paths_config.get("reports_dir", "reports") / "stage8"
    models_dir = project_root / paths_config.get("models_dir", "models")

    splits_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Calculate Dataset SHA-256 Hash for Lineage Verification
    dataset_sha256 = compute_file_sha256(feature_dataset_path)

    target_config = TargetConfig(
        source_column=target_config_raw.get("source_column", "groundwater_level"),
        group_column=target_config_raw.get("group_column", "District LGD Code"),
        date_column=target_config_raw.get("date_column", "Date"),
        forecast_horizon_days=int(target_config_raw.get("forecast_horizon_days", 7)),
        transform=target_config_raw.get("transform", "identity"),
        target_column_name=target_config_raw.get("target_column_name", "target"),
    )
    split_ratios = SplitRatios(
        train=float(split_config_raw.get("train_ratio", 0.7)),
        validation=float(split_config_raw.get("validation_ratio", 0.15)),
        test=float(split_config_raw.get("test_ratio", 0.15)),
    )

    df = load_feature_dataset(feature_dataset_path)

    df_with_target, transform_metadata = generate_forecast_target(df, target_config)

    target_stats = validate_target(df_with_target, target_config.target_column_name)

    exclude_columns = frozenset(
        {
            target_config.group_column,
            target_config.date_column,
            target_config.target_column_name,
        }
    )
    leakage_warnings = run_leakage_audit(
        df_with_target, target_config.target_column_name, exclude_columns
    )

    temporal_stats = run_temporal_continuity_check(
        df_with_target, target_config.group_column, target_config.date_column
    )

    train_df, validation_df, test_df, split_report = compute_time_based_split(
        df_with_target,
        target_config.date_column,
        target_config.group_column,
        target_config.target_column_name,
        split_ratios,
    )

    train_path = splits_dir / "train.parquet"
    validation_path = splits_dir / "validation.parquet"
    test_path = splits_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata_path = models_dir / "target_transform_metadata.json"
    save_transform_metadata(transform_metadata, metadata_path)

    validation_report = {
        "version": REPORT_VERSION,
        "stage": PIPELINE_STAGE,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_dataset_path": str(feature_dataset_path),
        "dataset_sha256": dataset_sha256,
        "target_config": {
            "source_column": target_config.source_column,
            "group_column": target_config.group_column,
            "date_column": target_config.date_column,
            "forecast_horizon_days": target_config.forecast_horizon_days,
            "transform": target_config.transform,
            "target_column_name": target_config.target_column_name,
        },
        "target_stats": target_stats,
        "leakage_audit_warnings": leakage_warnings,
        "temporal_continuity": temporal_stats,
        "split": split_report,
        "output_paths": {
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
            "target_transform_metadata": str(metadata_path),
        },
    }

    report_path = reports_dir / "dataset_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(validation_report, fh, indent=2)
    logger.info("build_training_dataset.report_saved", extra={"path": str(report_path)})

    validation_report["report_path"] = str(report_path)
    return validation_report


def main() -> None:
    """Entry point for standalone script execution."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    project_root = _project_root()
    try:
        result = build_training_dataset(project_root)
    except (TrainingDatasetBuildError, TargetEngineeringError) as exc:
        print(f"\n================ EXCEPTION TRACEBACK ================\n")
        traceback.print_exc()
        print(f"\n================ ERROR MESSAGE =====================")
        print(f"ERROR DETAILS: {exc}")
        print(f"====================================================\n")
        logger.exception("build_training_dataset.failed")
        sys.exit(1)
    except Exception as exc:
        print(f"\n================ UNEXPECTED EXCEPTION ================\n")
        traceback.print_exc()
        print(f"======================================================\n")
        logger.exception("build_training_dataset.failed_unexpected")
        sys.exit(1)

    logger.info("build_training_dataset.succeeded")
    logger.info(f"Dataset SHA-256: {result['dataset_sha256']}")
    logger.info(f"Train rows: {result['split']['train_rows']}")
    logger.info(f"Validation rows: {result['split']['validation_rows']}")
    logger.info(f"Test rows: {result['split']['test_rows']}")
    if result["leakage_audit_warnings"]:
        logger.warning(
            f"Leakage audit flagged {len(result['leakage_audit_warnings'])} feature(s) for review"
        )


if __name__ == "__main__":
    main()
