"""
===============================================================================
WATER INTELLIGENCE PLATFORM - STAGE 11 MODEL LOADER
Module: src/inference/model_loader.py
===============================================================================

LAYER: Inference
PURPOSE:
    Thread-safe singleton responsible for loading, validating, and exposing
    the frozen Stage 10 artifacts required for inference:

        - champion_model.joblib
        - feature_schema.json
        - metadata.json
        - selection_report.json

    Loading occurs exactly once per process (or lazily on first access if
    configured). The loaded artifacts are parsed into frozen, immutable
    dataclasses (FeatureSchema, ModelMetadata, SelectionReport) and exposed
    only through read-only properties.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import joblib
from sklearn.base import BaseEstimator

from src.common.logging_utils import LogTimer, get_logger, log_call
from src.inference.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ModelLoaderError,
    ModelNotLoadedError,
)
from src.inference.settings import Stage11Settings, get_settings

logger = get_logger("inference.model_loader")

# ---------------------------------------------------------------------------
# Artifact key constants
# ---------------------------------------------------------------------------
ARTIFACT_KEY_CHAMPION_MODEL = "champion_model"
ARTIFACT_KEY_FEATURE_SCHEMA = "feature_schema"
ARTIFACT_KEY_METADATA = "metadata"
ARTIFACT_KEY_SELECTION_REPORT = "selection_report"

HASH_VERIFIED_ARTIFACT_KEYS: tuple[str, ...] = (
    ARTIFACT_KEY_CHAMPION_MODEL,
    ARTIFACT_KEY_FEATURE_SCHEMA,
    ARTIFACT_KEY_SELECTION_REPORT,
)

SUPPORTED_STAGE10_ARTIFACT_VERSIONS: tuple[str, ...] = ("10.0.0",)

ALLOWED_EXTENSIONS: dict[str, tuple[str, ...]] = {
    ARTIFACT_KEY_CHAMPION_MODEL: (".joblib", ".pkl"),
    ARTIFACT_KEY_FEATURE_SCHEMA: (".json",),
    ARTIFACT_KEY_METADATA: (".json",),
    ARTIFACT_KEY_SELECTION_REPORT: (".json",),
}

# ---------------------------------------------------------------------------
# JSON field name constants
# ---------------------------------------------------------------------------
FIELD_FEATURE_NAMES = "feature_names"
FIELD_FEATURE_COUNT = "feature_count"
FIELD_TARGET_COLUMN = "target_column"
FIELD_MODEL_NAME = "model_name"
FIELD_MODEL_TYPE = "model_type"
FIELD_STAGE_VERSION = "stage_version"
FIELD_TRAINED_AT = "trained_at"
FIELD_ARTIFACT_HASHES = "artifact_hashes"
FIELD_CHAMPION_MODEL_NAME = "champion_model_name"
FIELD_SELECTION_METRIC = "selection_metric"
FIELD_SELECTION_SCORE = "selection_score"

REQUIRED_FEATURE_SCHEMA_KEYS: frozenset[str] = frozenset(
    {FIELD_FEATURE_NAMES, FIELD_FEATURE_COUNT, FIELD_TARGET_COLUMN}
)
REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        FIELD_MODEL_NAME,
        FIELD_MODEL_TYPE,
        FIELD_STAGE_VERSION,
        FIELD_TRAINED_AT,
        FIELD_ARTIFACT_HASHES,
    }
)
REQUIRED_SELECTION_REPORT_KEYS: frozenset[str] = frozenset(
    {FIELD_CHAMPION_MODEL_NAME, FIELD_SELECTION_METRIC, FIELD_SELECTION_SCORE}
)

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_HASH_CHUNK_SIZE_BYTES = 1024 * 1024


# ---------------------------------------------------------------------------
# Immutable artifact containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Immutable, validated representation of feature_schema.json."""

    feature_names: tuple[str, ...]
    feature_count: int
    target_column: str


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Immutable, validated representation of metadata.json."""

    model_name: str
    model_type: str
    stage_version: str
    trained_at: str
    artifact_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Immutable, validated representation of selection_report.json."""

    champion_model_name: str
    selection_metric: str
    selection_score: float


# ---------------------------------------------------------------------------
# Validation and sanity helpers
# ---------------------------------------------------------------------------
def _require_keys(
    raw: dict[str, Any], required: frozenset[str], artifact_key: str
) -> None:
    """Raise ArtifactIntegrityError if *raw* is missing any key in *required*."""
    missing = sorted(required - raw.keys())
    if missing:
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' is missing required key(s): {missing}",
            context={"artifact": artifact_key, "missing_keys": missing},
        )


def _require_non_empty_string(value: Any, field_name: str, artifact_key: str) -> str:
    """Raise ArtifactIntegrityError unless *value* is a non-empty string."""
    if not isinstance(value, str) or value.strip() == "":
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{field_name}' must be a non-empty string, got {value!r}",
            context={
                "artifact": artifact_key,
                "field": field_name,
                "value": repr(value),
            },
        )
    return value


def _require_valid_model_identifier(
    value: str, field_name: str, artifact_key: str
) -> str:
    """Raise ArtifactIntegrityError unless *value* is a valid identifier."""
    if not _MODEL_NAME_PATTERN.match(value):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{field_name}' is not a valid identifier: {value!r}",
            context={"artifact": artifact_key, "field": field_name, "value": value},
        )
    return value


def _normalize_model_identifier(value: str) -> str:
    """Normalize a model identifier for strict comparison."""
    return value.strip().casefold()


def _verify_artifact_file_sanity(path: Path, artifact_key: str) -> None:
    """Refinement 8: Strict pre-loading path, permission, size, and suffix check."""
    logger.debug("Verifying path integrity for artifact '%s' at %s", artifact_key, path)

    if not path.exists():
        raise ArtifactNotFoundError(
            f"Required artifact '{artifact_key}' does not exist at {path}",
            context={"artifact": artifact_key, "path": str(path)},
        )
    if not path.is_file():
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' path is not a regular file: {path}",
            context={"artifact": artifact_key, "path": str(path)},
        )
    if not os.access(path, os.R_OK):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' at {path} is not readable (permission denied).",
            context={"artifact": artifact_key, "path": str(path)},
        )
    if path.stat().st_size == 0:
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' at {path} is an empty file (0 bytes).",
            context={"artifact": artifact_key, "path": str(path)},
        )

    expected_suffixes = ALLOWED_EXTENSIONS.get(artifact_key, ())
    if expected_suffixes and path.suffix.lower() not in expected_suffixes:
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' has unexpected file extension '{path.suffix}'. Expected one of: {expected_suffixes}",
            context={
                "artifact": artifact_key,
                "path": str(path),
                "extension": path.suffix,
            },
        )


def _load_json_artifact(path: Path, artifact_key: str) -> dict[str, Any]:
    """Read and parse a JSON artifact file into a dictionary."""
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            raw = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"Failed to parse JSON artifact '{artifact_key}' at {path}: {exc}",
            context={"artifact": artifact_key, "path": str(path)},
        ) from exc

    if not isinstance(raw, dict):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' must contain a root JSON object, got {type(raw).__name__}",
            context={"artifact": artifact_key, "path": str(path)},
        )
    return raw


def _compute_sha256(path: Path) -> str:
    """Stream SHA-256 computation in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(_HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, artifact_key: str, expected_hash: str) -> None:
    """Verify calculated hash matches metadata recording exactly."""
    computed_hash = _compute_sha256(path)
    if computed_hash != expected_hash:
        raise ArtifactIntegrityError(
            f"SHA-256 mismatch for artifact '{artifact_key}': expected {expected_hash}, computed {computed_hash}",
            context={
                "artifact": artifact_key,
                "path": str(path),
                "expected_hash": expected_hash,
                "computed_hash": computed_hash,
            },
        )


# ---------------------------------------------------------------------------
# Structural Parsing Logic
# ---------------------------------------------------------------------------
def _parse_feature_schema(raw: dict[str, Any], artifact_key: str) -> FeatureSchema:
    """Parse and validate feature schema. Includes Case-Insensitive Duplicate Checking (Refinement 6)."""
    _require_keys(raw, REQUIRED_FEATURE_SCHEMA_KEYS, artifact_key)

    raw_names = raw[FIELD_FEATURE_NAMES]
    if not isinstance(raw_names, list) or len(raw_names) == 0:
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{FIELD_FEATURE_NAMES}' must be a non-empty list",
            context={"artifact": artifact_key},
        )

    feature_names: list[str] = []
    seen_normalized: set[str] = set()
    duplicates: set[str] = set()

    for index, name in enumerate(raw_names):
        if not isinstance(name, str) or name.strip() == "":
            raise ArtifactIntegrityError(
                f"Artifact '{artifact_key}' {FIELD_FEATURE_NAMES}[{index}] must be a non-empty string",
                context={"artifact": artifact_key, "index": index},
            )

        normalized_name = name.strip().casefold()
        if normalized_name in seen_normalized:
            duplicates.add(name.strip())
        seen_normalized.add(normalized_name)
        feature_names.append(name.strip())

    if duplicates:
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' contains duplicate feature name(s) (case-insensitive check): {sorted(duplicates)}",
            context={"artifact": artifact_key, "duplicates": sorted(duplicates)},
        )

    feature_count = raw[FIELD_FEATURE_COUNT]
    if not isinstance(feature_count, int) or isinstance(feature_count, bool):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{FIELD_FEATURE_COUNT}' must be an integer",
            context={"artifact": artifact_key, "value": repr(feature_count)},
        )

    if feature_count != len(feature_names):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' {FIELD_FEATURE_COUNT} ({feature_count}) does not match len({FIELD_FEATURE_NAMES}) ({len(feature_names)})",
            context={
                "artifact": artifact_key,
                "declared_count": feature_count,
                "actual_count": len(feature_names),
            },
        )

    target_column = _require_non_empty_string(
        raw[FIELD_TARGET_COLUMN], FIELD_TARGET_COLUMN, artifact_key
    )

    return FeatureSchema(
        feature_names=tuple(feature_names),
        feature_count=feature_count,
        target_column=target_column,
    )


def _parse_metadata(raw: dict[str, Any], artifact_key: str) -> ModelMetadata:
    """Parse and validate metadata.json."""
    _require_keys(raw, REQUIRED_METADATA_KEYS, artifact_key)

    model_name = _require_valid_model_identifier(
        _require_non_empty_string(
            raw[FIELD_MODEL_NAME], FIELD_MODEL_NAME, artifact_key
        ),
        FIELD_MODEL_NAME,
        artifact_key,
    )
    model_type = _require_valid_model_identifier(
        _require_non_empty_string(
            raw[FIELD_MODEL_TYPE], FIELD_MODEL_TYPE, artifact_key
        ),
        FIELD_MODEL_TYPE,
        artifact_key,
    )
    stage_version = _require_non_empty_string(
        raw[FIELD_STAGE_VERSION], FIELD_STAGE_VERSION, artifact_key
    )
    trained_at = _require_non_empty_string(
        raw[FIELD_TRAINED_AT], FIELD_TRAINED_AT, artifact_key
    )

    raw_hashes = raw[FIELD_ARTIFACT_HASHES]
    if not isinstance(raw_hashes, dict):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{FIELD_ARTIFACT_HASHES}' must be a JSON object",
            context={"artifact": artifact_key},
        )

    validated_hashes: dict[str, str] = {}
    for hash_key in HASH_VERIFIED_ARTIFACT_KEYS:
        hash_val = raw_hashes.get(hash_key)
        if not isinstance(hash_val, str) or not _SHA256_HEX_PATTERN.match(
            hash_val.strip().lower()
        ):
            raise ArtifactIntegrityError(
                f"Artifact '{artifact_key}' missing or invalid SHA-256 hash digest for '{hash_key}'",
                context={"artifact": artifact_key, "hash_key": hash_key},
            )
        validated_hashes[hash_key] = hash_val.strip().lower()

    return ModelMetadata(
        model_name=model_name,
        model_type=model_type,
        stage_version=stage_version,
        trained_at=trained_at,
        artifact_hashes=MappingProxyType(validated_hashes),
    )


def _parse_selection_report(raw: dict[str, Any], artifact_key: str) -> SelectionReport:
    """Parse and validate selection_report.json."""
    _require_keys(raw, REQUIRED_SELECTION_REPORT_KEYS, artifact_key)

    champion_model_name = _require_valid_model_identifier(
        _require_non_empty_string(
            raw[FIELD_CHAMPION_MODEL_NAME], FIELD_CHAMPION_MODEL_NAME, artifact_key
        ),
        FIELD_CHAMPION_MODEL_NAME,
        artifact_key,
    )
    selection_metric = _require_non_empty_string(
        raw[FIELD_SELECTION_METRIC], FIELD_SELECTION_METRIC, artifact_key
    )

    raw_score = raw[FIELD_SELECTION_SCORE]
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{FIELD_SELECTION_SCORE}' must be numeric",
            context={"artifact": artifact_key, "value": repr(raw_score)},
        )

    score = float(raw_score)
    if math.isnan(score) or math.isinf(score):
        raise ArtifactIntegrityError(
            f"Artifact '{artifact_key}' field '{FIELD_SELECTION_SCORE}' must be finite",
            context={"artifact": artifact_key, "value": score},
        )

    return SelectionReport(
        champion_model_name=champion_model_name,
        selection_metric=selection_metric,
        selection_score=score,
    )


# ---------------------------------------------------------------------------
# Estimator and Consistency Verifications
# ---------------------------------------------------------------------------
def _verify_cross_artifact_consistency(
    metadata: ModelMetadata, selection_report: SelectionReport
) -> None:
    """Verify model identifiers across metadata and selection report."""
    if _normalize_model_identifier(metadata.model_type) != _normalize_model_identifier(
        selection_report.champion_model_name
    ):
        raise ArtifactIntegrityError(
            "Cross-artifact integrity failed: metadata.model_type does not match selection_report.champion_model_name",
            context={
                "metadata_model_type": metadata.model_type,
                "selection_report_champion_model_name": selection_report.champion_model_name,
            },
        )


def _verify_version_compatibility(
    metadata: ModelMetadata, supported_versions: tuple[str, ...]
) -> None:
    """Refinement 1: Verify version compatibility using dynamic settings list."""
    if metadata.stage_version not in supported_versions:
        raise ModelLoaderError(
            f"Stage 10 artifact version '{metadata.stage_version}' is unsupported. Supported versions: {supported_versions}",
            context={
                "artifact_version": metadata.stage_version,
                "supported_versions": supported_versions,
            },
        )


def _load_champion_model(path: Path) -> BaseEstimator:
    """Refinement 5: Load champion estimator and verify presence of both predict and fit contracts."""
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Failed to unpickle champion model artifact at {path}: {exc}",
            context={"path": str(path)},
        ) from exc

    if not isinstance(model, BaseEstimator):
        raise ArtifactIntegrityError(
            f"Loaded champion object is not a scikit-learn BaseEstimator (got {type(model).__name__})",
            context={"path": str(path), "loaded_type": type(model).__name__},
        )

    # Refinement 5: Enhanced Model Contract Validation
    predict_fn = getattr(model, "predict", None)
    fit_fn = getattr(model, "fit", None)

    if predict_fn is None or not callable(predict_fn):
        raise ArtifactIntegrityError(
            f"Loaded model '{type(model).__name__}' does not expose a callable predict() method",
            context={"path": str(path), "loaded_type": type(model).__name__},
        )
    if fit_fn is None or not callable(fit_fn):
        raise ArtifactIntegrityError(
            f"Loaded model '{type(model).__name__}' does not expose a callable fit() method",
            context={"path": str(path), "loaded_type": type(model).__name__},
        )

    return model


def _verify_model_feature_count(
    model: BaseEstimator, feature_schema: FeatureSchema
) -> None:
    """Cross-verify estimator n_features_in_ with feature schema count."""
    expected_features = getattr(model, "n_features_in_", None)
    if (
        expected_features is not None
        and int(expected_features) != feature_schema.feature_count
    ):
        raise ArtifactIntegrityError(
            f"Champion model expects {int(expected_features)} features, but feature_schema declares {feature_schema.feature_count}",
            context={
                "model_n_features_in": int(expected_features),
                "schema_feature_count": feature_schema.feature_count,
            },
        )


# ---------------------------------------------------------------------------
# ModelLoader Singleton
# ---------------------------------------------------------------------------
class ModelLoader:
    """Thread-safe singleton managing Stage 10 inference artifacts."""

    _instance: ModelLoader | None = None
    _instance_lock: threading.Lock = threading.Lock()

    _load_lock: threading.Lock
    _is_loaded: bool
    _champion_model: BaseEstimator | None
    _feature_schema: FeatureSchema | None
    _metadata: ModelMetadata | None
    _selection_report: SelectionReport | None
    _settings: Stage11Settings | None

    def __new__(cls) -> ModelLoader:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._load_lock = threading.Lock()
                    instance._is_loaded = False
                    instance._champion_model = None
                    instance._feature_schema = None
                    instance._metadata = None
                    instance._selection_report = None
                    instance._settings = None
                    cls._instance = instance
                    logger.debug("ModelLoader singleton instance created.")
        return cls._instance

    @classmethod
    @log_call
    def get_instance(cls) -> ModelLoader:
        """Accessor for ModelLoader singleton."""
        return cls()

    @log_call
    def load(self, settings: Stage11Settings | None = None) -> None:
        """Load, parse, and validate all Stage 10 artifacts with phase-level timing."""
        if self._is_loaded:
            logger.debug("Artifacts already loaded; skipping redundant execution.")
            return

        with self._load_lock:
            if self._is_loaded:
                return

            resolved_settings = settings if settings is not None else get_settings()
            self._settings = resolved_settings

            with LogTimer(logger, "model_loader.total_load_sequence"):
                self._load_locked(resolved_settings)

    def _load_locked(self, settings: Stage11Settings) -> None:
        """Internal discovery and parsing execution phase."""
        artifacts_config = settings.artifacts

        # Refinement 7: Extensible Discovery Map
        artifact_paths: dict[str, Path] = {
            key: artifacts_config.get_artifact_path(key)
            for key in (
                ARTIFACT_KEY_CHAMPION_MODEL,
                ARTIFACT_KEY_FEATURE_SCHEMA,
                ARTIFACT_KEY_METADATA,
                ARTIFACT_KEY_SELECTION_REPORT,
            )
        }

        # Phase 1: Path & File Integrity Sanity Verification (Refinement 8)
        with LogTimer(logger, "model_loader.phase_1_file_sanity_check"):
            for artifact_key, artifact_path in artifact_paths.items():
                _verify_artifact_file_sanity(artifact_path, artifact_key)

        # Phase 2: JSON Parsing and Structural Validation (Refinement 2)
        with LogTimer(logger, "model_loader.phase_2_json_parsing"):
            feature_schema_raw = _load_json_artifact(
                artifact_paths[ARTIFACT_KEY_FEATURE_SCHEMA], ARTIFACT_KEY_FEATURE_SCHEMA
            )
            metadata_raw = _load_json_artifact(
                artifact_paths[ARTIFACT_KEY_METADATA], ARTIFACT_KEY_METADATA
            )
            selection_report_raw = _load_json_artifact(
                artifact_paths[ARTIFACT_KEY_SELECTION_REPORT],
                ARTIFACT_KEY_SELECTION_REPORT,
            )

            feature_schema = _parse_feature_schema(
                feature_schema_raw, ARTIFACT_KEY_FEATURE_SCHEMA
            )
            metadata = _parse_metadata(metadata_raw, ARTIFACT_KEY_METADATA)
            selection_report = _parse_selection_report(
                selection_report_raw, ARTIFACT_KEY_SELECTION_REPORT
            )

        # Phase 3: Cross-Artifact & Version Compatibility Checks
        if artifacts_config.verify_artifacts:
            with LogTimer(
                logger, "model_loader.phase_3_cross_artifact_and_version_checks"
            ):
                _verify_cross_artifact_consistency(metadata, selection_report)

                # Safely attempt resolution from settings if added in the future; fallback to module constant
                compatibility_cfg = getattr(settings, "compatibility", None)
                if compatibility_cfg is not None:
                    supported_versions = tuple(
                        getattr(
                            compatibility_cfg,
                            "supported_stage10_versions",
                            SUPPORTED_STAGE10_ARTIFACT_VERSIONS,
                        )
                    )
                else:
                    supported_versions = SUPPORTED_STAGE10_ARTIFACT_VERSIONS

                _verify_version_compatibility(metadata, supported_versions)

        # Phase 4: SHA-256 Digest Verification (Refinement 2)
        if artifacts_config.verify_hashes:
            with LogTimer(logger, "model_loader.phase_4_hash_verification"):
                for hash_key in HASH_VERIFIED_ARTIFACT_KEYS:
                    _verify_hash(
                        path=artifact_paths[hash_key],
                        artifact_key=hash_key,
                        expected_hash=metadata.artifact_hashes[hash_key],
                    )

        # Phase 5: Estimator Object Loading & Validation (Refinement 2 & 5)
        with LogTimer(logger, "model_loader.phase_5_model_loading"):
            champion_model = _load_champion_model(
                artifact_paths[ARTIFACT_KEY_CHAMPION_MODEL]
            )
            _verify_model_feature_count(champion_model, feature_schema)

        # Commit State
        self._feature_schema = feature_schema
        self._metadata = metadata
        self._selection_report = selection_report
        self._champion_model = champion_model
        self._is_loaded = True

        logger.info(
            "ModelLoader successfully initialized champion model '%s' (%d features).",
            metadata.model_name,
            feature_schema.feature_count,
        )

    # -----------------------------------------------------------------------
    # State & Internal Helpers
    # -----------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """Return loading status."""
        return self._is_loaded

    @log_call
    def clear(self) -> None:
        """Reset singleton state. Strictly for test isolation."""
        with self._load_lock:
            self._champion_model = None
            self._feature_schema = None
            self._metadata = None
            self._selection_report = None
            self._settings = None
            self._is_loaded = False
            logger.debug("ModelLoader singleton state reset.")

    def _ensure_loaded(self) -> None:
        """Refinement 4: Check loading state or trigger auto-load if enabled in settings."""
        if self._is_loaded:
            return

        settings = self._settings or get_settings()
        auto_load = getattr(settings.artifacts, "auto_load_on_access", False)

        if auto_load:
            logger.info(
                "Auto-load triggered on property access (settings.artifacts.auto_load_on_access=True)."
            )
            self.load(settings)
        else:
            raise ModelNotLoadedError(
                "ModelLoader has not loaded artifacts yet. Call load() prior to accessing properties.",
                context={"is_loaded": False},
            )

    # -----------------------------------------------------------------------
    # Artifact Accessors (Refinement 3: Hot-path @log_call removed)
    # -----------------------------------------------------------------------
    @property
    def champion_model(self) -> BaseEstimator:
        """Return the unpickled champion model estimator."""
        self._ensure_loaded()
        assert self._champion_model is not None
        return self._champion_model

    @property
    def feature_schema(self) -> FeatureSchema:
        """Return the immutable FeatureSchema."""
        self._ensure_loaded()
        assert self._feature_schema is not None
        return self._feature_schema

    @property
    def metadata(self) -> ModelMetadata:
        """Return the immutable ModelMetadata."""
        self._ensure_loaded()
        assert self._metadata is not None
        return self._metadata

    @property
    def selection_report(self) -> SelectionReport:
        """Return the immutable SelectionReport."""
        self._ensure_loaded()
        assert self._selection_report is not None
        return self._selection_report

    # -----------------------------------------------------------------------
    # Convenience Shortcut Properties
    # -----------------------------------------------------------------------
    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names tuple."""
        return self.feature_schema.feature_names

    @property
    def feature_count(self) -> int:
        """Return total input feature count."""
        return self.feature_schema.feature_count

    @property
    def target_column(self) -> str:
        """Return target column identifier."""
        return self.feature_schema.target_column
