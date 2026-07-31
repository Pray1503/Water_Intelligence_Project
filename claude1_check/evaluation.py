"""Stage 10 - Layer 5: Model Evaluation.

Evaluates every successfully trained model from a Layer 4 TrainingReport
against the VALIDATION split only. Never touches the test split -- test
evaluation is explicitly out of scope for model selection (frozen rule 15)
and belongs to a later, separate stage.

No champion selection happens here -- that is Layer 6 (selection.py).
This layer's sole responsibility is producing per-model metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.stage10.common import (
    ExecutionMode,
    compute_regression_metrics,
    extract_features_and_target,
)
from src.stage10.dataset_bundle import DatasetBundle
from src.stage10.exceptions import EvaluationError
from src.stage10.logging_utils import log_call
from src.stage10.training import TrainedModelResult, TrainingReport
from src.stage10.validation import ValidationConfig

logger = logging.getLogger("stage10")


@dataclass(frozen=True)
class ModelEvaluation:
    """Validation-split evaluation result for a single model.

    Attributes
    ----------
    name:
        Model name (matches the key in TrainingReport.results).
    rmse, mae, r2:
        Regression metrics computed on the validation split.
    validation_rows:
        Number of validation target rows over which metrics were computed.
    prediction_duration_seconds:
        Wall-clock time spent inside model.predict().
    evaluated_at:
        UTC timestamp recorded when evaluation completed.
    """

    name: str
    rmse: float
    mae: float
    r2: float
    validation_rows: int
    prediction_duration_seconds: float
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate result of evaluating every model in a TrainingReport.

    Attributes
    ----------
    results:
        Mapping of model name -> ModelEvaluation, preserving the training
        insertion order for every model that evaluated successfully.
    failed_models:
        Mapping of model name -> error message (formatted with exception type),
        for models that failed evaluation. Always empty in STRICT mode.
    mode:
        The ExecutionMode this evaluation run used.
    total_duration_seconds:
        Wall-clock time for the entire evaluate_all_models call.
    evaluated_at:
        UTC timestamp recorded when the evaluation run completed.
    """

    results: dict[str, ModelEvaluation]
    failed_models: dict[str, str]
    mode: ExecutionMode
    total_duration_seconds: float
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@log_call
def evaluate_single_model(
    result: TrainedModelResult,
    bundle: DatasetBundle,
) -> ModelEvaluation:
    """Evaluate one trained model against the validation split.

    Parameters
    ----------
    result:
        A TrainedModelResult from Layer 4, supplying the fitted model
        and the exact feature/target columns it was trained on.
    bundle:
        DatasetBundle supplying bundle.validation. bundle.test is never
        referenced by this function or anything it calls.

    Returns
    -------
    ModelEvaluation

    Raises
    ------
    EvaluationError
        If validation set is empty, model lacks predict(), prediction fails,
        shape mismatch occurs, or predictions contain NaN/Inf values.
    """
    if bundle.validation.empty:
        raise EvaluationError(
            f"Cannot evaluate model '{result.name}': validation dataset split is empty."
        )

    if not hasattr(result.model, "predict") or not callable(
        getattr(result.model, "predict")
    ):
        raise EvaluationError(
            f"Model '{result.name}' of type '{type(result.model).__name__}' "
            f"does not implement a callable 'predict()' method."
        )

    X_val, y_val = extract_features_and_target(
        bundle.validation, result.feature_columns, result.target_column
    )

    start_time = time.perf_counter()
    try:
        y_pred = result.model.predict(X_val)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed for model '%s'", result.name)
        raise EvaluationError(
            f"Model '{result.name}' failed to predict on {len(X_val)} "
            f"validation rows: {exc}"
        ) from exc
    duration = time.perf_counter() - start_time

    y_pred_array = np.asarray(y_pred, dtype=float)

    if len(y_pred_array) != len(y_val):
        raise EvaluationError(
            f"Model '{result.name}' prediction length mismatch: "
            f"expected {len(y_val)} predictions to match target rows, "
            f"got {len(y_pred_array)}."
        )

    if not np.all(np.isfinite(y_pred_array)):
        n_bad = int((~np.isfinite(y_pred_array)).sum())
        raise EvaluationError(
            f"Model '{result.name}' produced {n_bad} NaN/Inf prediction(s) "
            f"out of {len(y_pred_array)} on the validation split."
        )

    rmse, mae, r2 = compute_regression_metrics(y_val, y_pred_array)

    logger.info(
        "Evaluated model '%s': rmse=%.6f mae=%.6f r2=%.6f (%s rows, %.4fs)",
        result.name,
        rmse,
        mae,
        r2,
        len(y_val),
        duration,
    )

    return ModelEvaluation(
        name=result.name,
        rmse=rmse,
        mae=mae,
        r2=r2,
        validation_rows=len(y_val),
        prediction_duration_seconds=duration,
    )


@log_call
def evaluate_all_models(
    training_report: TrainingReport,
    bundle: DatasetBundle,
    validation_config: ValidationConfig,
    mode: ExecutionMode = ExecutionMode.STRICT,
) -> EvaluationReport:
    """Evaluate every model in *training_report* against the validation split.

    Parameters
    ----------
    training_report:
        Layer 4 output containing one TrainedModelResult per successfully
        trained model.
    bundle:
        DatasetBundle already validated by Layer 2. Only bundle.validation
        is used -- bundle.test is never touched, per frozen rule 15.
    validation_config:
        Maintained for API interface consistency across stage layers.
    mode:
        STRICT (default): the first evaluation failure raises immediately.
        BEST_EFFORT: failing models are recorded in
        EvaluationReport.failed_models and evaluation continues for the
        rest.

    Returns
    -------
    EvaluationReport

    Raises
    ------
    EvaluationError
        If training_report.results is empty, if any model fails under
        STRICT mode, or if BEST_EFFORT mode results in zero successful
        evaluations.
    """
    _ = validation_config

    if not training_report.results:
        raise EvaluationError(
            "training_report.results is empty; there are no trained models to evaluate."
        )

    overall_start = time.perf_counter()
    results: dict[str, ModelEvaluation] = {}
    failed_models: dict[str, str] = {}

    for name, trained_result in training_report.results.items():
        try:
            results[name] = evaluate_single_model(trained_result, bundle)
        except Exception as exc:
            if mode is ExecutionMode.STRICT:
                if isinstance(exc, EvaluationError):
                    raise
                raise EvaluationError(
                    f"Unexpected failure during evaluation of model '{name}': {exc}"
                ) from exc

            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "BEST_EFFORT: skipping model '%s' after failure: %s", name, err_msg
            )
            failed_models[name] = err_msg

    if not results:
        raise EvaluationError(
            f"All {len(training_report.results)} model(s) failed evaluation "
            f"in BEST_EFFORT mode; nothing succeeded for Layer 6 to select from."
        )

    total_duration = time.perf_counter() - overall_start

    report = EvaluationReport(
        results=results,
        failed_models=failed_models,
        mode=mode,
        total_duration_seconds=total_duration,
    )

    logger.info(
        "Layer 5 evaluation complete: %d succeeded, %d failed, mode=%s, %.4fs",
        len(results),
        len(failed_models),
        mode.value,
        total_duration,
    )

    return report
