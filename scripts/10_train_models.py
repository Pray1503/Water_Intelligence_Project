#!/usr/bin/env python3
"""Stage 10 Orchestration Script: Model Training Entry Point.

Glues together existing Stage 10 modules without modifying them:
1. Ensures project root is in sys.path for direct script execution.
2. Loads Stage 9 dataset bundle via Layer 1 API (load_dataset_bundle).
3. Loads preprocessing metadata with fallback handling for target_column.
4. Builds ValidationConfig and runs validate_dataset_bundle via Layer 2 API.
5. Loads model configs and populates ModelRegistry via Layer 3 API.
6. Executes train_all_models via Layer 4 API.
7. Evaluates trained models via Layer 5 API (evaluate_all_models).
8. Selects champion model via Layer 6 API (select_champion_model).
9. Persists champion artifacts via Layer 7 API (persist_stage10_artifacts).
10. Displays the required summary report.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the Python path when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Any

from src.stage10.dataset_bundle import load_dataset_bundle
from src.stage10.evaluation import evaluate_all_models
from src.stage10.exceptions import Stage10Error
from src.stage10.model_registry import (
    build_model_registry,
    load_model_configs_from_dict,
)
from src.stage10.persistence import persist_stage10_artifacts
from src.stage10.selection import select_champion_model
from src.stage10.training import train_all_models
from src.stage10.validation import (
    ValidationConfig,
    validate_dataset_bundle,
)

# Output directory for persisted champion artifacts
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "stage10"

# Configure project logger
logger = logging.getLogger("stage10")


def print_banner(title: str) -> None:
    """Print a prominent CLI banner."""
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}\n")


def find_config_file(project_root: Path) -> Path:
    """Locate the project configuration file across standard paths."""
    candidate_paths = [
        project_root / "config" / "stage10.yaml",
        project_root / "config" / "config.yaml",
        project_root / "config" / "settings.yaml",
        project_root / "stage10.yaml",
        project_root / "config.yaml",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not locate a valid configuration YAML file in {project_root}"
    )


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def find_metadata_file(project_root: Path) -> Path:
    """Locate Stage 9 preprocessing metadata JSON file."""
    candidates = [
        project_root / "models" / "preprocessing_metadata.json",
        project_root / "data" / "preprocessed" / "preprocessing_metadata.json",
        project_root / "data" / "processed" / "preprocessing_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Preprocessing metadata JSON file not found in candidates: {candidates}"
    )


def main() -> int:
    """Orchestrate Stage 10 model training workflow.

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    print_banner("STAGE 10: MODEL TRAINING ORCHESTRATION")

    try:
        data_dir = PROJECT_ROOT / "data" / "preprocessed"

        # ------------------------------------------------------------------
        # 1. Load & Parse Preprocessing Metadata
        # ------------------------------------------------------------------
        metadata_path = find_metadata_file(PROJECT_ROOT)
        logger.info("Loading Stage 9 preprocessing metadata from %s...", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata_dict = json.load(fh)

        # Fallback handling for target_column
        target_column = metadata_dict.get("target_column")
        if target_column is None:
            logger.warning(
                "target_column not found in preprocessing metadata. "
                "Using default target column 'target'."
            )
            target_column = "target"

        feature_columns = metadata_dict.get(
            "feature_columns",
            metadata_dict.get("feature_names_out"),
        )
        if not feature_columns:
            raise Stage10Error("No feature columns found in preprocessing metadata.")

        preprocessing_hash = metadata_dict.get(
            "preprocessing_hash",
            metadata_dict.get("pipeline_sha256"),
        )

        # ------------------------------------------------------------------
        # 2. Load Dataset Bundle via Layer 1 API
        # ------------------------------------------------------------------
        logger.info("Loading preprocessed dataset bundle via Layer 1 API...")
        train_path = data_dir / "train.parquet"
        validation_path = data_dir / "validation.parquet"
        test_path = data_dir / "test.parquet"

        bundle = load_dataset_bundle(
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
        )

        # ------------------------------------------------------------------
        # 3. Build ValidationConfig & Validate Bundle via Layer 2 API
        # ------------------------------------------------------------------
        logger.info("Configuring Layer 2 schema validation...")
        validation_config = ValidationConfig(
            target_column=target_column,
            feature_columns=feature_columns,
            preprocessing_metadata_path=metadata_path,
            expected_preprocessing_hash=preprocessing_hash,
        )

        logger.info("Executing Layer 2 dataset bundle validation...")
        validate_dataset_bundle(bundle, validation_config)
        logger.info("DatasetBundle validation passed successfully.")

        # ------------------------------------------------------------------
        # 4. Load Model Configuration & Build Registry via Layer 3 API
        # ------------------------------------------------------------------
        config_path = find_config_file(PROJECT_ROOT)
        logger.info("Loading model configuration from %s...", config_path)
        raw_config_dict = load_yaml_config(config_path)

        stage10_cfg = raw_config_dict.get("stage10", raw_config_dict)
        raw_model_configs = stage10_cfg.get("models", [])

        if not raw_model_configs:
            raise Stage10Error("No model configurations found under 'models' key.")

        model_configs = load_model_configs_from_dict(raw_model_configs)

        logger.info("Building Layer 3 Model Registry...")
        model_registry = build_model_registry(model_configs)

        # ------------------------------------------------------------------
        # 5. Train Models via Layer 4 API
        # ------------------------------------------------------------------
        logger.info("Executing Layer 4 Model Training...")
        training_report = train_all_models(
            model_registry=model_registry,
            bundle=bundle,
            validation_config=validation_config,
        )

        # ------------------------------------------------------------------
        # 6. Evaluate Trained Models via Layer 5 API (validation split only)
        # ------------------------------------------------------------------
        logger.info("Executing Layer 5 Model Evaluation...")
        evaluation_report = evaluate_all_models(
            training_report=training_report,
            bundle=bundle,
            validation_config=validation_config,
        )

        # ------------------------------------------------------------------
        # 7. Select Champion Model via Layer 6 API
        # ------------------------------------------------------------------
        logger.info("Executing Layer 6 Champion Selection...")
        selection_report = select_champion_model(evaluation_report)

        # ------------------------------------------------------------------
        # 8. Persist Champion Artifacts via Layer 7 API
        # ------------------------------------------------------------------
        logger.info("Executing Layer 7 Artifact Persistence...")
        persistence_report = persist_stage10_artifacts(
            training_report=training_report,
            evaluation_report=evaluation_report,
            selection_report=selection_report,
            validation_config=validation_config,
            output_dir=MODEL_OUTPUT_DIR,
        )

        # ------------------------------------------------------------------
        # 9. Print Orchestration Summary Report
        # ------------------------------------------------------------------
        print_banner("Stage 10 Training Complete")

        trained_names = list(training_report.results.keys())
        first_result = next(iter(training_report.results.values()))
        champion_name = selection_report.champion.champion_name
        champion_eval = selection_report.champion.champion_evaluation

        print("Models trained:")
        for name in trained_names:
            print(f"  - {name}")

        print(
            f"\nTraining time:\n  {training_report.total_duration_seconds:.4f} seconds"
        )

        print(f"\nRows used:\n  {first_result.train_rows:,}")

        print(f"\nFeatures:\n  {len(first_result.feature_columns)}")

        print(f"\nChampion model:\n  {champion_name}")

        print(
            f"\nChampion validation metrics:\n"
            f"  rmse={champion_eval.rmse:.6f}  "
            f"mae={champion_eval.mae:.6f}  "
            f"r2={champion_eval.r2:.6f}"
        )

        if selection_report.champion.tied_with:
            print(
                f"  (tied with {selection_report.champion.tied_with}, "
                "won by insertion order)"
            )

        print(f"\nArtifacts persisted to:\n  {persistence_report.output_dir}")
        print(f"  - {persistence_report.champion_model_path}")
        print(f"  - {persistence_report.metadata_path}")
        print(f"  - {persistence_report.evaluation_report_path}")
        print(f"  - {persistence_report.selection_report_path}")
        print(f"  - {persistence_report.feature_schema_path}")
        print(f"  - {persistence_report.version_info_path}")

        print("\n" + "-" * 40 + "\n")
        return 0

    except Stage10Error as err:
        logger.error("Stage 10 Validation/Execution Error: %s", err)
        print(f"\n[ERROR] Stage 10 Pipeline Failure: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        logger.exception("Unexpected error occurred during Stage 10 execution.")
        print(f"\n[ERROR] Unexpected System Failure: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(main())
