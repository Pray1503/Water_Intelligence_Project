"""
===============================================================================
WATER INTELLIGENCE PLATFORM - INFERENCE EXCEPTION HIERARCHY
Module: src/inference/exceptions.py
===============================================================================

LAYER: Inference
PURPOSE:
    Defines every custom exception raised by the Stage 11 inference pipeline
    (settings, model_loader, validator, transformer, risk_engine, prediction_engine).
    All exceptions inherit from Stage11Error (which in turn inherits from
    Stage10Error), so the API layer (Step 11, routes.py) can catch every
    inference-layer failure via a single base exception when mapping to an
    HTTP response, while still discriminating on the specific subclass when
    a different status code is warranted.

ARCHITECTURAL NOTES:
    - Fail Fast: every exception here represents a condition the pipeline
      refuses to proceed past -- there is no "log and continue" path in
      Stage 11's inference flow.
    - Contextual Errors: Stage11Error extends Stage10Error by adding an optional
      context dictionary without mutating Stage10Error's constructor signature.
    - Exception Chaining: every raise site elsewhere in Stage 11 is expected
      to use `raise <ThisModuleException>(...) from exc` when converting a
      lower-level exception into one of these, so the original cause is never lost.
    - No Bare Exception: nothing in this module is ever raised or caught
      with a bare `except:`.

IMPORTS:
    - src.stage10.exceptions.Stage10Error
"""

from __future__ import annotations

from typing import Any

from src.stage10.exceptions import Stage10Error

__all__ = [
    "Stage11Error",
    "ConfigurationError",
    "ConfigurationNotFoundError",
    "ConfigurationValidationError",
    "ModelLoaderError",
    "ArtifactNotFoundError",
    "ArtifactIntegrityError",
    "ModelNotLoadedError",
    "RequestValidationError",
    "FeatureValidationError",
    "SchemaValidationError",
    "DataQualityError",
    "TransformationError",
    "RiskEngineError",
    "RiskThresholdConfigurationError",
    "PredictionEngineError",
]


class Stage11Error(Stage10Error):
    """Base exception for every Stage 11 inference-pipeline failure.

    Inherits directly from Stage10Error to maintain a single, unbroken exception
    hierarchy across all platform stages. Extends Stage10Error by supporting a
    structured context dictionary for enriched logging and HTTP response details.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{super().__str__()} | context={self.context}"
        return super().__str__()


# ---------------------------------------------------------------------------
# Configuration Loader (Step 4 - settings.py)
# ---------------------------------------------------------------------------


class ConfigurationError(Stage11Error):
    """Base exception for Stage 11 configuration loading and validation failures."""


class ConfigurationNotFoundError(ConfigurationError):
    """Raised when config/stage11.yaml cannot be located at the expected path."""


class ConfigurationValidationError(ConfigurationError):
    """Raised when config/stage11.yaml fails schema validation against Stage11Settings."""


# ---------------------------------------------------------------------------
# Model Loader (Step 5 - model_loader.py)
# ---------------------------------------------------------------------------


class ModelLoaderError(Stage11Error):
    """Base exception for failures loading Stage 10 artifacts
    (champion_model.joblib, feature_schema.json, metadata.json,
    selection_report.json) from models/stage10/."""


class ArtifactNotFoundError(ModelLoaderError):
    """Raised when a required Stage 10 artifact file does not exist at its
    expected path under models/stage10/."""


class ArtifactIntegrityError(ModelLoaderError):
    """Raised when a loaded Stage 10 artifact fails an integrity check
    (e.g. a computed hash does not match the hash recorded in metadata.json)."""


class ModelNotLoadedError(ModelLoaderError):
    """Raised when inference code attempts to access the champion model or
    feature schema before the model loader singleton has completed loading."""


# ---------------------------------------------------------------------------
# Validator (Step 6 - validator.py)
# ---------------------------------------------------------------------------


class RequestValidationError(Stage11Error):
    """Base exception for a prediction request that fails Stage 11 validation.

    Represents a client error (HTTP 400 / 422), distinct from server-side errors.
    """


class FeatureValidationError(RequestValidationError):
    """Intermediate base exception for individual feature value or vector
    validation failures. Reusable by Stage 12 (Decision Simulator) and
    Stage 13 (AI Copilot)."""


class SchemaValidationError(FeatureValidationError):
    """Raised when a request's feature vector does not structurally match
    feature_schema.json (wrong count, unknown key, or missing required key)."""


class DataQualityError(FeatureValidationError):
    """Raised when feature values fail content checks (e.g., NaN/null values or
    unsupported data types)."""


# ---------------------------------------------------------------------------
# Transformer (Step 7 - transformer.py)
# ---------------------------------------------------------------------------


class TransformationError(Stage11Error):
    """Raised when a validated request cannot be aligned into the exact array
    format required by champion_model.joblib."""


# ---------------------------------------------------------------------------
# Risk Engine (Step 8 - risk_engine.py)
# ---------------------------------------------------------------------------


class RiskEngineError(Stage11Error):
    """Base exception for failures translating model outputs into risk classifications."""


class RiskThresholdConfigurationError(RiskEngineError):
    """Raised when config/stage11.yaml's risk thresholds are missing or malformed."""


# ---------------------------------------------------------------------------
# Prediction Engine (Step 9 - prediction_engine.py)
# ---------------------------------------------------------------------------


class PredictionEngineError(Stage11Error):
    """Raised for unexpected orchestration failures within prediction_engine.py itself."""
