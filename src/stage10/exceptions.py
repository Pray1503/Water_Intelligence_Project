"""Stage 10 - Custom Domain Exceptions.

Provides custom exception classes for Stage 10 pipeline components.
All custom exceptions inherit from Stage10Error so callers may catch
all Stage 10 failures via a single base exception when appropriate.
"""

from __future__ import annotations


class Stage10Error(Exception):
    """Base exception for all Stage 10 pipeline operations."""


# ---------------------------------------------------------------------
# Layer 1 - Dataset Bundle
# ---------------------------------------------------------------------


class DatasetBundleError(Stage10Error):
    """Raised when a DatasetBundle cannot be created or is invalid."""


class DatasetLoadError(DatasetBundleError):
    """Raised when a Stage 9 parquet split file cannot be located or
    read while assembling a DatasetBundle."""


# ---------------------------------------------------------------------
# Layer 2 - Validation
# ---------------------------------------------------------------------


class ValidationError(Stage10Error):
    """Raised when dataset validation fails."""


class DataValidationError(ValidationError):
    """Raised when dataset contents violate validation rules."""


class MetadataValidationError(ValidationError):
    """Raised when preprocessing metadata validation fails."""


class SchemaValidationError(ValidationError):
    """Raised when a dataset split's schema (columns, ordering, target
    presence) does not match what ValidationConfig expects."""


# ---------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------


class FeatureExtractionError(Stage10Error):
    """Raised when feature and target extraction from a DataFrame fails."""


# ---------------------------------------------------------------------
# Layer 3 - Model Registry
# ---------------------------------------------------------------------


class ModelRegistryError(Stage10Error):
    """Raised when model registry construction or model creation fails."""


class UnknownModelError(ModelRegistryError):
    """Raised when a ModelConfig names a model that is not in
    MODEL_BUILDER_REGISTRY."""


# ---------------------------------------------------------------------
# Layer 4 - Training
# ---------------------------------------------------------------------


class TrainingError(Stage10Error):
    """Raised when model training fails."""


# ---------------------------------------------------------------------
# Layer 5 - Evaluation
# ---------------------------------------------------------------------


class EvaluationError(Stage10Error):
    """Raised when model evaluation fails due to prediction errors,
    invalid prediction values, metric computation failures,
    or evaluation workflow failures."""


# ---------------------------------------------------------------------
# Layer 6 - Selection
# ---------------------------------------------------------------------


class SelectionError(Stage10Error):
    """Raised when champion model selection fails."""


# ---------------------------------------------------------------------
# Layer 7 - Persistence
# ---------------------------------------------------------------------


class PersistenceError(Stage10Error):
    """Raised when saving or loading Stage 10 artifacts fails."""
