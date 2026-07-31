"""Stage 10 - Layer 7: Artifact Persistence.

Persists the outputs of Layers 4-6 once a champion has been selected:

    * the fitted champion model
    * a curated metadata document (model type, hyperparameters, training
      provenance, champion selection outcome)
    * the full Layer 5 EvaluationReport
    * the full Layer 6 SelectionReport
    * the champion's feature schema (target column, ordered feature list)
    * version information (Python, scikit-learn, and optional
      xgboost/lightgbm versions, plus platform details)

This layer reads Layer 4/5/6 outputs and writes files. It never
retrains, re-evaluates, or re-selects anything.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.stage10.common import ExecutionMode
from src.stage10.evaluation import EvaluationReport
from src.stage10.exceptions import PersistenceError
from src.stage10.logging_utils import log_call
from src.stage10.selection import SelectionReport
from src.stage10.training import TrainedModelResult, TrainingReport
from src.stage10.validation import ValidationConfig

logger = logging.getLogger("stage10")

# ---------------------------------------------------------------------------
# Fixed, deterministic artifact filenames. Every persistence run writes
# the same set of files under the same names, so downstream stages (and
# a human inspecting the output directory) never have to guess.
# ---------------------------------------------------------------------------
CHAMPION_MODEL_FILENAME = "champion_model.joblib"
METADATA_FILENAME = "metadata.json"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"
SELECTION_REPORT_FILENAME = "selection_report.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
VERSION_INFO_FILENAME = "version_info.json"


@dataclass(frozen=True)
class PersistenceReport:
    """Record of every artifact this layer wrote to disk.

    Attributes
    ----------
    output_dir:
        Directory all artifacts were written under.
    champion_name:
        Name of the persisted champion model.
    champion_model_path, metadata_path, evaluation_report_path,
    selection_report_path, feature_schema_path, version_info_path:
        Absolute paths to each written artifact.
    failed_artifacts:
        Mapping of artifact label -> error message, for artifacts that
        could not be written. Always empty in STRICT mode (the first
        failure raises immediately instead of being recorded here).
    mode:
        The ExecutionMode this persistence run used.
    total_duration_seconds:
        Wall-clock time for the entire persist_stage10_artifacts call.
    persisted_at:
        UTC timestamp recorded when persistence completed.
    """

    output_dir: Path
    champion_name: str
    champion_model_path: Path | None
    metadata_path: Path | None
    evaluation_report_path: Path | None
    selection_report_path: Path | None
    feature_schema_path: Path | None
    version_info_path: Path | None
    failed_artifacts: dict[str, str]
    mode: ExecutionMode
    total_duration_seconds: float
    persisted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _json_default(obj: Any) -> Any:
    """Fallback serializer for json.dump covering every non-primitive
    type that appears inside Stage 10's report dataclasses."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(data: Any, path: Path, artifact_label: str) -> None:
    """Write *data* to *path* as pretty-printed, deterministic JSON.

    Parameters
    ----------
    data:
        JSON-serializable value (dataclasses, datetimes, Paths, and
        Enums are handled automatically via _json_default).
    path:
        Destination file path. Parent directories are created if needed.
    artifact_label:
        Human-readable label used in log messages and error text.

    Raises
    ------
    PersistenceError
        If the parent directory cannot be created or the file cannot be
        written/serialized.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, default=_json_default)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write %s to %s", artifact_label, path)
        raise PersistenceError(
            f"Failed to write {artifact_label} to '{path}': {exc}"
        ) from exc

    logger.debug("Wrote %s -> %s", artifact_label, path)


def _resolve_champion_result(
    training_report: TrainingReport, champion_name: str
) -> TrainedModelResult:
    """Look up the champion's TrainedModelResult in *training_report*.

    Raises
    ------
    PersistenceError
        If *champion_name* is not present in training_report.results --
        this indicates Layer 6 selected a model that Layer 4 never
        trained, which should be impossible but is checked explicitly
        rather than assumed.
    """
    champion_result = training_report.results.get(champion_name)
    if champion_result is None:
        raise PersistenceError(
            f"Selected champion '{champion_name}' was not found in "
            f"training_report.results (available: "
            f"{sorted(training_report.results)}). Cannot persist a model "
            "that was never trained."
        )
    return champion_result


@log_call
def save_champion_model(
    training_report: TrainingReport,
    champion_name: str,
    output_dir: Path,
) -> Path:
    """Persist the champion's fitted model via joblib.

    Parameters
    ----------
    training_report:
        Layer 4 output containing the fitted model for every
        successfully trained candidate.
    champion_name:
        Name of the model Layer 6 selected as champion.
    output_dir:
        Directory to write CHAMPION_MODEL_FILENAME under.

    Returns
    -------
    Path to the written model file.

    Raises
    ------
    PersistenceError
        If the champion is not found in training_report, or if the
        model cannot be serialized to disk.
    """
    champion_result = _resolve_champion_result(training_report, champion_name)
    output_path = output_dir / CHAMPION_MODEL_FILENAME

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion_result.model, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save champion model '%s'", champion_name)
        raise PersistenceError(
            f"Failed to save champion model '{champion_name}' "
            f"(type={type(champion_result.model).__name__}) to "
            f"'{output_path}': {exc}"
        ) from exc

    logger.info(
        "Saved champion model '%s' (type=%s) -> %s",
        champion_name,
        type(champion_result.model).__name__,
        output_path,
    )
    return output_path


def _extract_hyperparameters(model: Any) -> dict[str, Any]:
    """Best-effort extraction of a fitted model's hyperparameters via
    scikit-learn's get_params() convention.

    Returns an empty dict rather than raising if the model does not
    expose get_params(), since this information is diagnostic, not
    load-bearing -- a missing get_params() must not block persistence
    of the model itself.
    """
    get_params = getattr(model, "get_params", None)
    if not callable(get_params):
        return {}
    try:
        params = get_params()
    except Exception:  # noqa: BLE001
        logger.debug(
            "get_params() raised for model type %s; recording empty "
            "hyperparameters instead of failing persistence.",
            type(model).__name__,
        )
        return {}

    # Ensure every value is JSON-serializable; anything that isn't
    # (nested estimator objects, callables, etc.) is stringified rather
    # than dropped, so the metadata document stays complete.
    safe_params: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            safe_params[key] = value
        else:
            safe_params[key] = repr(value)
    return safe_params


@log_call
def build_metadata(
    training_report: TrainingReport,
    selection_report: SelectionReport,
    validation_config: ValidationConfig,
) -> dict[str, Any]:
    """Build the curated Layer 7 metadata document for the champion.

    Deliberately reconstructs a plain dict (rather than json-dumping
    TrainedModelResult directly) because TrainedModelResult.model holds
    the actual fitted estimator object, which is never JSON-serializable
    and must never be attempted here.

    Parameters
    ----------
    training_report:
        Layer 4 output.
    selection_report:
        Layer 6 output; selection_report.champion.champion_name
        identifies which trained model to describe.
    validation_config:
        The ValidationConfig used for this run, for provenance
        (target column, preprocessing metadata path/hash).

    Returns
    -------
    A JSON-serializable metadata dict.

    Raises
    ------
    PersistenceError
        If the champion is not found in training_report.
    """
    champion_name = selection_report.champion.champion_name
    champion_result = _resolve_champion_result(training_report, champion_name)
    champion_evaluation = selection_report.champion.champion_evaluation

    metadata = {
        "champion_name": champion_name,
        "model_type": type(champion_result.model).__name__,
        "hyperparameters": _extract_hyperparameters(champion_result.model),
        "target_column": champion_result.target_column,
        "feature_count": len(champion_result.feature_columns),
        "train_rows": champion_result.train_rows,
        "training_duration_seconds": champion_result.training_duration_seconds,
        "trained_at": champion_result.trained_at,
        "validation_metrics": {
            "rmse": champion_evaluation.rmse,
            "mae": champion_evaluation.mae,
            "r2": champion_evaluation.r2,
            "validation_rows": champion_evaluation.validation_rows,
        },
        "tied_with": selection_report.champion.tied_with,
        "candidates_ranked": len(selection_report.ranking),
        "excluded_models": selection_report.excluded_models,
        "selection_mode": selection_report.mode,
        "selected_at": selection_report.champion.selected_at,
        "preprocessing_metadata_path": validation_config.preprocessing_metadata_path,
        "expected_preprocessing_hash": validation_config.expected_preprocessing_hash,
    }

    logger.debug("Built metadata document for champion '%s'", champion_name)
    return metadata


@log_call
def build_feature_schema(
    training_report: TrainingReport,
    champion_name: str,
    validation_config: ValidationConfig,
) -> dict[str, Any]:
    """Build the champion's feature schema document.

    Cross-checks the champion's actual training feature order against
    validation_config.feature_columns when the latter is non-empty, so
    a drift between what Layer 2 validated and what Layer 4 actually
    trained on is caught here rather than shipped silently.

    Parameters
    ----------
    training_report:
        Layer 4 output.
    champion_name:
        Name of the selected champion model.
    validation_config:
        The ValidationConfig used for this run.

    Returns
    -------
    A JSON-serializable feature schema dict.

    Raises
    ------
    PersistenceError
        If the champion is not found in training_report, or if
        validation_config.feature_columns is non-empty and does not
        exactly match the champion's trained feature order.
    """
    champion_result = _resolve_champion_result(training_report, champion_name)

    if (
        validation_config.feature_columns
        and list(validation_config.feature_columns) != champion_result.feature_columns
    ):
        raise PersistenceError(
            f"Feature schema drift detected for champion '{champion_name}': "
            f"ValidationConfig.feature_columns "
            f"({len(validation_config.feature_columns)} columns) does not "
            f"match the champion's actual trained feature_columns "
            f"({len(champion_result.feature_columns)} columns). Refusing "
            "to persist a feature schema that does not match what was "
            "actually trained."
        )

    schema = {
        "target_column": champion_result.target_column,
        "feature_count": len(champion_result.feature_columns),
        "feature_columns": champion_result.feature_columns,
    }

    logger.debug(
        "Built feature schema for champion '%s': %d feature(s)",
        champion_name,
        len(champion_result.feature_columns),
    )
    return schema


def _get_optional_package_version(package_name: str) -> str | None:
    """Return an installed package's __version__, or None if it is not
    installed. Used for optional dependencies (xgboost, lightgbm) that
    model_registry.py itself treats as optional."""
    try:
        module = __import__(package_name)
    except Exception:  # Covers ImportError and missing environment C-extension errors
        return None
    return getattr(module, "__version__", "unknown")


@log_call
def build_version_info() -> dict[str, Any]:
    """Build the version/environment information document.

    Returns
    -------
    A JSON-serializable dict of interpreter, platform, and library
    versions. Optional libraries (xgboost, lightgbm) that are not
    installed are recorded as None rather than raising, matching
    model_registry.py's own treatment of them as optional.
    """
    version_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "xgboost_version": _get_optional_package_version("xgboost"),
        "lightgbm_version": _get_optional_package_version("lightgbm"),
        "captured_at": datetime.now(timezone.utc),
    }

    logger.debug("Captured version info: %s", version_info)
    return version_info


@log_call
def persist_stage10_artifacts(
    training_report: TrainingReport,
    evaluation_report: EvaluationReport,
    selection_report: SelectionReport,
    validation_config: ValidationConfig,
    output_dir: Path,
    mode: ExecutionMode = ExecutionMode.STRICT,
) -> PersistenceReport:
    """Persist every Layer 7 artifact for the selected champion.

    Writes, under *output_dir*:
        * champion_model.joblib   -- the fitted champion model
        * metadata.json           -- curated training/selection metadata
        * evaluation_report.json  -- the full Layer 5 EvaluationReport
        * selection_report.json   -- the full Layer 6 SelectionReport
        * feature_schema.json     -- the champion's feature schema
        * version_info.json       -- environment/library versions

    Parameters
    ----------
    training_report:
        Layer 4 output.
    evaluation_report:
        Layer 5 output.
    selection_report:
        Layer 6 output; selection_report.champion.champion_name
        identifies the model to persist.
    validation_config:
        The ValidationConfig used for this run.
    output_dir:
        Directory to write all artifacts under. Created if it does not
        exist.
    mode:
        STRICT (default): the first artifact-write failure raises
        immediately, leaving whatever was already written on disk.
        BEST_EFFORT: a failing artifact is recorded in
        PersistenceReport.failed_artifacts and the remaining artifacts
        are still attempted.

    Returns
    -------
    PersistenceReport

    Raises
    ------
    PersistenceError
        If the champion is not found in training_report, if any
        artifact fails to write under STRICT mode, or if every artifact
        fails to write under BEST_EFFORT mode.
    """
    overall_start = time.perf_counter()
    output_dir = Path(output_dir)
    champion_name = selection_report.champion.champion_name

    logger.info(
        "Layer 7 persistence starting for champion '%s' -> %s (mode=%s)",
        champion_name,
        output_dir,
        mode.value,
    )

    # Fail fast if the champion doesn't exist in training_report at all --
    # every artifact below depends on this, so there is no point
    # attempting any of them individually under BEST_EFFORT if this
    # fails; it indicates a real integration problem between Layers 4
    # and 6, not a per-artifact write failure.
    _resolve_champion_result(training_report, champion_name)

    artifact_paths: dict[str, Path | None] = {
        "champion_model": None,
        "metadata": None,
        "evaluation_report": None,
        "selection_report": None,
        "feature_schema": None,
        "version_info": None,
    }
    failed_artifacts: dict[str, str] = {}

    def _attempt(label: str, fn) -> None:
        """Run fn() (which must return the Path it wrote), storing the
        result in artifact_paths[label] on success. On failure: raise
        under STRICT, or record into failed_artifacts under BEST_EFFORT.
        """
        try:
            artifact_paths[label] = fn()
        except Exception as exc:  # noqa: BLE001
            if mode is ExecutionMode.STRICT:
                if isinstance(exc, PersistenceError):
                    raise
                raise PersistenceError(
                    f"Unexpected failure while persisting artifact " f"'{label}': {exc}"
                ) from exc
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "BEST_EFFORT: failed to persist artifact '%s': %s", label, err_msg
            )
            failed_artifacts[label] = err_msg

    def _save_champion_model_artifact() -> Path:
        return save_champion_model(training_report, champion_name, output_dir)

    def _save_metadata_artifact() -> Path:
        path = output_dir / METADATA_FILENAME
        metadata = build_metadata(training_report, selection_report, validation_config)
        _write_json(metadata, path, "metadata")
        return path

    def _save_evaluation_report_artifact() -> Path:
        path = output_dir / EVALUATION_REPORT_FILENAME
        _write_json(evaluation_report, path, "evaluation report")
        return path

    def _save_selection_report_artifact() -> Path:
        path = output_dir / SELECTION_REPORT_FILENAME
        _write_json(selection_report, path, "selection report")
        return path

    def _save_feature_schema_artifact() -> Path:
        path = output_dir / FEATURE_SCHEMA_FILENAME
        schema = build_feature_schema(training_report, champion_name, validation_config)
        _write_json(schema, path, "feature schema")
        return path

    def _save_version_info_artifact() -> Path:
        path = output_dir / VERSION_INFO_FILENAME
        _write_json(build_version_info(), path, "version info")
        return path

    _attempt("champion_model", _save_champion_model_artifact)
    _attempt("metadata", _save_metadata_artifact)
    _attempt("evaluation_report", _save_evaluation_report_artifact)
    _attempt("selection_report", _save_selection_report_artifact)
    _attempt("feature_schema", _save_feature_schema_artifact)
    _attempt("version_info", _save_version_info_artifact)

    if mode is ExecutionMode.BEST_EFFORT and len(failed_artifacts) == len(
        artifact_paths
    ):
        raise PersistenceError(
            f"All {len(artifact_paths)} artifact(s) failed to persist in "
            f"BEST_EFFORT mode for champion '{champion_name}'; nothing was "
            "written."
        )

    total_duration = time.perf_counter() - overall_start

    report = PersistenceReport(
        output_dir=output_dir,
        champion_name=champion_name,
        champion_model_path=artifact_paths["champion_model"],
        metadata_path=artifact_paths["metadata"],
        evaluation_report_path=artifact_paths["evaluation_report"],
        selection_report_path=artifact_paths["selection_report"],
        feature_schema_path=artifact_paths["feature_schema"],
        version_info_path=artifact_paths["version_info"],
        failed_artifacts=failed_artifacts,
        mode=mode,
        total_duration_seconds=total_duration,
    )

    logger.info(
        "Layer 7 persistence complete for champion '%s': %d/%d artifact(s) "
        "written, mode=%s, %.4fs",
        champion_name,
        len(artifact_paths) - len(failed_artifacts),
        len(artifact_paths),
        mode.value,
        total_duration,
    )

    return report
