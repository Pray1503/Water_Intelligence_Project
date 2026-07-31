"""
===============================================================================
WATER INTELLIGENCE PLATFORM - INFERENCE SETTINGS LOADER
Module: src/inference/settings.py
===============================================================================

LAYER: Inference / Infrastructure
PURPOSE:
    Provides typed, validated, immutable access to Stage 11 configuration
    defined in `config/stage11.yaml`. Converts raw YAML into frozen Pydantic models
    and manages settings caching via a thread-safe singleton wrapper.

ARCHITECTURAL NOTES:
    - Zero Schema Drift: Every Pydantic model in this file maps 1:1 to the frozen
      structure and key names of `config/stage11.yaml`.
    - Project-Root Path Resolution: DEFAULT_CONFIG_PATH is dynamically resolved relative
      to the repository root so invocation working directory differences (FastAPI, pytest,
      CLI scripts) never break configuration loading.
    - Explicit Artifact Properties: Exposes typed `.champion_model_path`, `.metadata_path`,
      `.feature_schema_path`, and `.selection_report_path` helpers directly on
      ArtifactPathsConfig.
    - Telemetry Consistency: Standard functions are wrapped with @log_call for
      automatic ENTRY/EXIT execution tracing matching Stage 10 observability patterns.
    - Fail Fast: Missing configuration files or invalid schema structure raises
      ConfigurationNotFoundError or ConfigurationValidationError immediately upon load.
    - Immutability: All Pydantic models use `frozen=True` and `extra="forbid"` to
      prevent runtime mutation or accidental key injection.

IMPORTS:
    - src.common.logging_utils.get_logger
    - src.common.logging_utils.log_call
    - src.inference.exceptions.ConfigurationNotFoundError
    - src.inference.exceptions.ConfigurationValidationError
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.common.logging_utils import get_logger, log_call
from src.inference.exceptions import (
    ConfigurationNotFoundError,
    ConfigurationValidationError,
)

logger = get_logger(__name__)

# Resolve project root dynamically (3 levels up from src/inference/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "stage11.yaml"


# ---------------------------------------------------------------------------
# Pydantic Configuration Models (Matching frozen config/stage11.yaml 1:1)
# ---------------------------------------------------------------------------


class Stage11MetaConfig(BaseModel):
    """Stage 11 Metadata Settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    environment: str


class ArtifactPathsConfig(BaseModel):
    """Artifact Directory & Filename Settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_directory: Path
    champion_model: str
    feature_schema: str
    metadata: str
    selection_report: str
    read_only: bool
    verify_artifacts: bool
    verify_hashes: bool

    def get_artifact_path(self, filename_key: str) -> Path:
        """Resolve a full Path for a named artifact relative to base_directory.

        Args:
            filename_key: Key name matching an artifact attribute
                          ('champion_model', 'feature_schema', 'metadata', 'selection_report').

        Returns:
            Resolved Path to the target artifact file.

        Raises:
            ValueError: If `filename_key` is not a known artifact key.
        """
        mapping = {
            "champion_model": self.champion_model,
            "feature_schema": self.feature_schema,
            "metadata": self.metadata,
            "selection_report": self.selection_report,
        }

        if filename_key not in mapping:
            raise ValueError(
                f"Unknown artifact key '{filename_key}'. "
                f"Available keys: {list(mapping.keys())}"
            )
        return self.base_directory / str(mapping[filename_key])

    @property
    def champion_model_path(self) -> Path:
        """Resolved Path to champion_model.joblib."""
        return self.get_artifact_path("champion_model")

    @property
    def feature_schema_path(self) -> Path:
        """Resolved Path to feature_schema.json."""
        return self.get_artifact_path("feature_schema")

    @property
    def metadata_path(self) -> Path:
        """Resolved Path to metadata.json."""
        return self.get_artifact_path("metadata")

    @property
    def selection_report_path(self) -> Path:
        """Resolved Path to selection_report.json."""
        return self.get_artifact_path("selection_report")


class ConsoleLoggingConfig(BaseModel):
    """Console Logger Sub-configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool


class LoggingConfig(BaseModel):
    """Platform Logging Configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    logger_name: str
    level: str
    propagate: bool
    console: ConsoleLoggingConfig
    formatter: str
    date_format: str

    @property
    def level_number(self) -> int:
        """Return standard logging integer level (e.g., DEBUG -> 10)."""
        level_name = self.level.upper()
        if hasattr(logging, level_name):
            val = getattr(logging, level_name)
            if isinstance(val, int):
                return val
        raise ValueError(
            f"Invalid logging level name '{self.level}'. Must be a standard logging level string."
        )


class RiskThresholdsConfig(BaseModel):
    """Risk Category Numerical Thresholds (Optional until calibration)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    very_low: float | None = Field(default=None, ge=0.0, le=1.0)
    low: float | None = Field(default=None, ge=0.0, le=1.0)
    moderate: float | None = Field(default=None, ge=0.0, le=1.0)
    high: float | None = Field(default=None, ge=0.0, le=1.0)
    critical: float | None = Field(default=None, ge=0.0, le=1.0)


class RiskEngineConfig(BaseModel):
    """Risk Classification Engine Settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold_source: str
    calibration_status: str
    labels: list[str]
    thresholds: RiskThresholdsConfig


class PredictionConfig(BaseModel):
    """Prediction Engine Execution Boundaries & Flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strict_validation: bool
    startup_validation: bool
    cache_model: bool
    return_model_metadata: bool
    enable_context: bool


class PerformanceConfig(BaseModel):
    """Inference Engine Performance & Execution Boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    singleton_model_loader: bool
    lazy_loading: bool
    model_cache: bool
    startup_validation: bool
    max_workers: int = Field(..., gt=0)


class DebugConfig(BaseModel):
    """Inference Debugging & Pipeline Tracing Flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    log_inputs: bool
    log_predictions: bool
    log_execution_time: bool
    log_validation: bool
    log_transformations: bool
    log_model_loading: bool
    include_exception_context: bool


class ApiEndpointsConfig(BaseModel):
    """API Endpoint Feature Flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    health: bool
    model_info: bool
    predict: bool


class ApiConfig(BaseModel):
    """FastAPI Service Routing Settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: str
    endpoints: ApiEndpointsConfig


class Stage11Settings(BaseModel):
    """Root Settings Schema for Stage 11."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage11: Stage11MetaConfig
    logging: LoggingConfig
    artifacts: ArtifactPathsConfig
    risk_engine: RiskEngineConfig
    prediction: PredictionConfig
    performance: PerformanceConfig
    debug: DebugConfig
    api: ApiConfig


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------


@log_call
def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Stage11Settings:
    """Load, parse, and validate Stage 11 YAML configuration file.

    Args:
        config_path: Path to `stage11.yaml`. Defaults to project root relative path.

    Returns:
        Validated `Stage11Settings` immutable instance.

    Raises:
        ConfigurationNotFoundError: If the YAML configuration file is missing.
        ConfigurationValidationError: If YAML structure or data type fails validation.
    """
    target_path = Path(config_path)
    logger.debug("Attempting to load configuration from: %s", target_path)

    if not target_path.is_file():
        logger.error("Configuration file not found: %s", target_path)
        raise ConfigurationNotFoundError(
            f"Stage 11 configuration file not found at '{target_path}'",
            context={"path": str(target_path)},
        )

    try:
        with target_path.open("r", encoding="utf-8") as f:
            raw_data: Any = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to parse YAML file at %s: %s", target_path, exc)
        raise ConfigurationValidationError(
            f"Failed to parse YAML configuration at '{target_path}': {exc}",
            context={"path": str(target_path)},
        ) from exc

    if not isinstance(raw_data, dict):
        logger.error("YAML root at '%s' is not a dictionary", target_path)
        raise ConfigurationValidationError(
            f"YAML root at '{target_path}' must be a dictionary, got {type(raw_data).__name__}",
            context={"path": str(target_path)},
        )

    try:
        settings = Stage11Settings.model_validate(raw_data)
        logger.info(
            "Stage 11 configuration successfully loaded [name=%s, version=%s, env=%s]",
            settings.stage11.name,
            settings.stage11.version,
            settings.stage11.environment,
        )
        return settings
    except ValidationError as exc:
        logger.error("Configuration validation failed for %s: %s", target_path, exc)
        raise ConfigurationValidationError(
            f"Configuration schema validation failed for '{target_path}'",
            context={"errors": exc.errors()},
        ) from exc


@log_call
@functools.lru_cache(maxsize=1)
def get_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Stage11Settings:
    """Thread-safe cached singleton accessor for Stage 11 settings."""
    return load_settings(config_path)


@log_call
def clear_settings_cache() -> None:
    """Clear the LRU cache for settings (useful during testing)."""
    get_settings.cache_clear()
    logger.info("Settings cache cleared.")
