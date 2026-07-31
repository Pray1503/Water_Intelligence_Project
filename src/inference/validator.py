"""
===============================================================================
WATER INTELLIGENCE PLATFORM - STAGE 11 INFERENCE VALIDATOR
Module: src/inference/validator.py
===============================================================================

LAYER: Inference
PURPOSE:
    Strictly validates raw prediction payloads against the loaded Stage 10
    FeatureSchema prior to feature vector construction or model evaluation.

    Guarantees:
        - Payload is a valid non-empty mapping.
        - All required schema features are present (no missing features).
        - No extra/unknown schema features are present.
        - Feature values are finite numeric types (float or int) and non-boolean.
        - Input payload is NEVER mutated or transformed.
        - Raises ONLY Stage11 exceptions (RequestValidationError,
          SchemaValidationError, DataQualityError).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.common.logging_utils import LogTimer, get_logger, log_call
from src.inference.exceptions import (
    DataQualityError,
    RequestValidationError,
    SchemaValidationError,
)
from src.inference.model_loader import FeatureSchema, ModelLoader

logger = get_logger("inference.validator")


class PayloadValidator:
    """Validator enforcing payload integrity against the model FeatureSchema."""

    def __init__(self, model_loader: ModelLoader | None = None) -> None:
        self._model_loader = model_loader or ModelLoader.get_instance()

    @log_call
    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate an incoming prediction feature payload without mutation.

        Args:
            payload: Raw mapping of feature names to values.

        Returns:
            The validated payload dictionary (unmutated shallow copy).

        Raises:
            RequestValidationError: If payload is not a mapping or is empty.
            SchemaValidationError: If missing features, unknown features,
                corrupted schema, or feature count mismatch occur.
            DataQualityError: If boolean, non-numeric, or non-finite values occur.
        """
        with LogTimer(logger, "validator.validate"):
            logger.debug("Starting payload validation.")

            if not isinstance(payload, Mapping):
                logger.warning(
                    "Validation failed: payload is not a mapping (%s)",
                    type(payload).__name__,
                )
                raise RequestValidationError(
                    f"Prediction payload must be a key-value mapping, got {type(payload).__name__}",
                    context={"received_type": type(payload).__name__},
                )

            if not payload:
                logger.warning("Validation failed: empty payload received.")
                raise RequestValidationError(
                    "Prediction payload cannot be empty",
                    context={"payload_len": 0},
                )

            schema: FeatureSchema = self._model_loader.feature_schema
            logger.debug(
                "Loaded feature schema containing %d features.",
                schema.feature_count,
            )

            expected_features: tuple[str, ...] = schema.feature_names
            if not expected_features:
                logger.error(
                    "Validation failed: loaded feature schema contains zero features."
                )
                raise SchemaValidationError(
                    "Feature schema contains zero features.",
                    context={"feature_count": 0},
                )

            expected_set = set(expected_features)
            payload_keys = set(payload.keys())

            # Check schema keys (missing / extra)
            missing_features = sorted(expected_set - payload_keys)
            if missing_features:
                logger.warning(
                    "Validation failed: missing %d feature(s)", len(missing_features)
                )
                raise SchemaValidationError(
                    f"Payload is missing required schema feature(s): {missing_features}",
                    context={
                        "missing_features": missing_features,
                        "expected_count": schema.feature_count,
                        "received_count": len(payload_keys),
                    },
                )

            unknown_features = sorted(payload_keys - expected_set)
            if unknown_features:
                logger.warning(
                    "Validation failed: detected %d unknown feature(s)",
                    len(unknown_features),
                )
                raise SchemaValidationError(
                    f"Payload contains unexpected extra feature(s): {unknown_features}",
                    context={
                        "unknown_features": unknown_features,
                        "expected_count": schema.feature_count,
                        "received_count": len(payload_keys),
                    },
                )

            if len(payload_keys) != schema.feature_count:
                logger.warning("Validation failed: feature count mismatch")
                raise SchemaValidationError(
                    f"Payload feature count mismatch: expected {schema.feature_count}, got {len(payload_keys)}",
                    context={
                        "expected_count": schema.feature_count,
                        "received_count": len(payload_keys),
                    },
                )

            # O(n) Single pass for Type & Finite Value checks
            invalid_types: dict[str, str] = {}
            non_finite_values: dict[str, Any] = {}

            for name in expected_features:
                val = payload[name]
                # Reject booleans (bool inherits from int in Python)
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    invalid_types[name] = type(val).__name__
                    continue

                float_val = float(val)
                if math.isnan(float_val) or math.isinf(float_val):
                    non_finite_values[name] = val

            if invalid_types:
                logger.warning(
                    "Validation failed: %d field(s) have invalid data types",
                    len(invalid_types),
                )
                raise DataQualityError(
                    f"Payload feature(s) must be numeric (int/float) and non-boolean: {invalid_types}",
                    context={"invalid_types": invalid_types},
                )

            if non_finite_values:
                logger.warning(
                    "Validation failed: %d field(s) contain non-finite numbers",
                    len(non_finite_values),
                )
                raise DataQualityError(
                    f"Payload feature(s) contain non-finite numeric values (NaN/Inf): {non_finite_values}",
                    context={"non_finite_values": non_finite_values},
                )

            logger.info("Payload validation completed successfully.")
            logger.debug(
                "Successfully validated payload with %d features.", schema.feature_count
            )
            return dict(payload)
