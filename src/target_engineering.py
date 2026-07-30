"""Stage 8 - Target Engineering.

Generates the forward-looking forecast target from the frozen feature
dataset, defines the forecast horizon, and produces/persists the metadata
required to invert any target transform back to original units in later
pipeline stages.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, get_args

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TransformName = Literal["identity", "log1p"]
SUPPORTED_TRANSFORMS: tuple[TransformName, ...] = get_args(TransformName)

METADATA_VERSION = "1.0.0"
PIPELINE_STAGE = "Stage8_TargetEngineering"


class TargetEngineeringError(Exception):
    """Raised when target engineering cannot proceed safely."""


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for forecast target generation.

    Attributes:
        source_column: Column in the feature dataset to forecast (e.g.
            "groundwater_level").
        group_column: Column identifying independent time series (e.g.
            "District LGD Code"). The target is never shifted across
            group boundaries.
        date_column: Column establishing chronological order within
            each group.
        forecast_horizon_days: Number of days ahead to forecast. Must
            be a positive integer.
        transform: Transform applied to source_column before shifting.
            "identity" leaves values unchanged; "log1p" applies
            log(1 + x) and requires non-negative source values.
        target_column_name: Name of the generated target column.
    """

    source_column: str
    group_column: str
    date_column: str
    forecast_horizon_days: int
    transform: TransformName = "identity"
    target_column_name: str = "target"

    def __post_init__(self) -> None:
        if self.forecast_horizon_days <= 0:
            raise TargetEngineeringError(
                f"forecast_horizon_days must be positive, got "
                f"{self.forecast_horizon_days}"
            )
        if not self.source_column:
            raise TargetEngineeringError("source_column must not be empty")
        if not self.group_column:
            raise TargetEngineeringError("group_column must not be empty")
        if not self.date_column:
            raise TargetEngineeringError("date_column must not be empty")
        if self.transform not in SUPPORTED_TRANSFORMS:
            raise TargetEngineeringError(
                f"Unsupported transform '{self.transform}'. Supported: {SUPPORTED_TRANSFORMS}"
            )


@dataclass(frozen=True)
class TargetTransformMetadata:
    """Inverse-transform metadata persisted alongside the model manifest.

    Attributes:
        transform: The transform that was applied to the source column.
        source_column: The original (untransformed) measurement column
            the target was derived from.
        forecast_horizon_days: The forecast horizon used to shift the
            target.
        version: Schema version of this metadata block.
        stage: Pipeline stage generating the metadata.
        created_timestamp: UTC ISO timestamp when metadata was generated.
    """

    transform: TransformName
    source_column: str
    forecast_horizon_days: int
    version: str = METADATA_VERSION
    stage: str = PIPELINE_STAGE
    created_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.transform not in SUPPORTED_TRANSFORMS:
            raise TargetEngineeringError(
                f"Invalid transform '{self.transform}' in metadata. "
                f"Supported: {SUPPORTED_TRANSFORMS}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict suitable for JSON persistence."""
        return asdict(self)


# =====================================================================
# Extensible Transform Dispatch
# =====================================================================

TransformFunc = Callable[[pd.Series], pd.Series]
InverseTransformFunc = Callable[[np.ndarray], np.ndarray]

_TRANSFORM_REGISTRY: dict[TransformName, TransformFunc] = {}
_INVERSE_TRANSFORM_REGISTRY: dict[TransformName, InverseTransformFunc] = {}


def register_transform(
    name: TransformName,
    forward_fn: TransformFunc,
    inverse_fn: InverseTransformFunc,
) -> None:
    """Register a new transform and its inverse for future extensibility."""
    _TRANSFORM_REGISTRY[name] = forward_fn
    _INVERSE_TRANSFORM_REGISTRY[name] = inverse_fn


# Default Registrations
def _identity_forward(series: pd.Series) -> pd.Series:
    return series


def _identity_inverse(values: np.ndarray) -> np.ndarray:
    return values


def _log1p_forward(series: pd.Series) -> pd.Series:
    if (series.dropna() < 0).any():
        raise TargetEngineeringError(
            "log1p transform requested but the source column contains "
            "negative values"
        )
    return np.log1p(series)


def _log1p_inverse(values: np.ndarray) -> np.ndarray:
    return np.expm1(values)


register_transform("identity", _identity_forward, _identity_inverse)
register_transform("log1p", _log1p_forward, _log1p_inverse)


def _apply_transform(series: pd.Series, transform: TransformName) -> pd.Series:
    """Apply the configured transform via registry dispatch."""
    logger.debug(
        "target_engineering.applying_transform",
        extra={"transform": transform, "rows": len(series)},
    )
    if transform not in _TRANSFORM_REGISTRY:
        raise TargetEngineeringError(
            f"Unknown transform '{transform}'. Registered: {list(_TRANSFORM_REGISTRY.keys())}"
        )
    return _TRANSFORM_REGISTRY[transform](series)


def _inverse_transform_values(
    values: np.ndarray, metadata: TargetTransformMetadata
) -> np.ndarray:
    """Invert a transform via registry dispatch."""
    if metadata.transform not in _INVERSE_TRANSFORM_REGISTRY:
        raise TargetEngineeringError(
            f"Unknown transform '{metadata.transform}' in metadata. "
            f"Registered: {list(_INVERSE_TRANSFORM_REGISTRY.keys())}"
        )
    return _INVERSE_TRANSFORM_REGISTRY[metadata.transform](values)


def inverse_transform_predictions(
    predictions: pd.Series | np.ndarray, metadata: TargetTransformMetadata
) -> np.ndarray:
    """Invert the target transform to return predictions to original units."""
    values = np.asarray(predictions, dtype=float)
    return _inverse_transform_values(values, metadata)


# =====================================================================
# Validation Helpers (Public for Unit Testing)
# =====================================================================


def validate_target_dataframe(df: pd.DataFrame, config: TargetConfig) -> None:
    """Public helper to validate input dataframe schemas, types, and constraints."""
    logger.debug(
        "target_engineering.validating_dataframe",
        extra={"total_rows": len(df), "total_cols": len(df.columns)},
    )

    required_columns = {config.source_column, config.group_column, config.date_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise TargetEngineeringError(
            f"Input dataframe missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise TargetEngineeringError(
            "Input dataframe is empty; cannot generate a target"
        )

    # Dtype Validation
    if not pd.api.types.is_numeric_dtype(df[config.source_column]):
        raise TargetEngineeringError(
            f"Source column '{config.source_column}' must be numeric, got {df[config.source_column].dtype}"
        )

    if pd.api.types.is_float_dtype(df[config.group_column]):
        raise TargetEngineeringError(
            f"Group column '{config.group_column}' should be an integer code or string/object ID, got float dtype"
        )


def validate_target_metadata(metadata: TargetTransformMetadata) -> None:
    """Public helper to validate target metadata integrity."""
    if metadata.transform not in SUPPORTED_TRANSFORMS:
        raise TargetEngineeringError(
            f"Invalid metadata transform '{metadata.transform}'"
        )
    if metadata.forecast_horizon_days <= 0:
        raise TargetEngineeringError("Metadata forecast_horizon_days must be positive")


# =====================================================================
# Main Target Generation Function
# =====================================================================


def generate_forecast_target(
    df: pd.DataFrame, config: TargetConfig
) -> tuple[pd.DataFrame, TargetTransformMetadata]:
    """Generate a forward-looking forecast target.

    For each group (e.g. district), the target at row t is the value of
    config.source_column at t + forecast_horizon_days, computed strictly
    from later rows within the same group.
    """
    logger.debug(
        "target_engineering.start",
        extra={
            "source_col": config.source_column,
            "group_col": config.group_column,
            "date_col": config.date_column,
            "horizon_days": config.forecast_horizon_days,
            "transform": config.transform,
        },
    )

    validate_target_dataframe(df, config)

    working = df.copy()

    # Dtype/Parse Date Handling
    working[config.date_column] = pd.to_datetime(
        working[config.date_column], errors="coerce"
    )
    n_unparseable_dates = int(working[config.date_column].isna().sum())
    if n_unparseable_dates:
        raise TargetEngineeringError(
            f"{n_unparseable_dates} rows have an unparseable "
            f"'{config.date_column}' value; cannot establish chronological order"
        )

    # Log NaN Statistics in Source Column
    raw_source_nans = int(working[config.source_column].isna().sum())
    nan_pct = (raw_source_nans / len(working)) * 100
    if raw_source_nans > 0:
        nan_by_group = (
            working[working[config.source_column].isna()]
            .groupby(config.group_column)
            .size()
            .to_dict()
        )
        logger.warning(
            "target_engineering.source_nans_detected",
            extra={
                "total_nans": raw_source_nans,
                "nan_percentage": f"{nan_pct:.2f}%",
                "affected_groups": len(nan_by_group),
                "nan_counts_per_group": nan_by_group,
            },
        )
    else:
        logger.debug("target_engineering.no_source_nans_detected")

    logger.debug("target_engineering.sorting_rows")
    working = working.sort_values(
        [config.group_column, config.date_column]
    ).reset_index(drop=True)

    # Duplicate Key Check with Diagnostic Breakdown
    dup_mask = working.duplicated(
        subset=[config.group_column, config.date_column], keep=False
    )
    if dup_mask.any():
        dup_samples = (
            working[dup_mask]
            .groupby([config.group_column, config.date_column])
            .size()
            .reset_index(name="count")
            .head(5)
            .to_dict(orient="records")
        )
        raise TargetEngineeringError(
            f"{int(dup_mask.sum())} rows share duplicate "
            f"({config.group_column}, {config.date_column}) keys. "
            f"Sample offending key combinations: {dup_samples}"
        )

    transformed_source = _apply_transform(
        working[config.source_column], config.transform
    )

    logger.debug("target_engineering.shifting_target_by_group")
    working[config.target_column_name] = transformed_source.groupby(
        working[config.group_column]
    ).shift(-config.forecast_horizon_days)

    n_missing_target = int(working[config.target_column_name].isna().sum())
    if n_missing_target:
        logger.warning(
            "target_engineering.horizon_truncation",
            extra={
                "rows_without_target": n_missing_target,
                "total_rows": int(len(working)),
                "forecast_horizon_days": config.forecast_horizon_days,
                "reason": "insufficient future data within group to shift target",
            },
        )

    metadata = TargetTransformMetadata(
        transform=config.transform,
        source_column=config.source_column,
        forecast_horizon_days=config.forecast_horizon_days,
    )
    validate_target_metadata(metadata)

    logger.info(
        "target_engineering.completed",
        extra={
            "source_column": config.source_column,
            "target_column": config.target_column_name,
            "group_column": config.group_column,
            "forecast_horizon_days": config.forecast_horizon_days,
            "transform": config.transform,
            "rows_in": int(len(df)),
            "rows_out": int(len(working)),
            "rows_with_valid_target": int(len(working) - n_missing_target),
        },
    )

    return working, metadata


# =====================================================================
# Persistence Logic
# =====================================================================


def save_transform_metadata(metadata: TargetTransformMetadata, path: Path) -> None:
    """Persist inverse-transform metadata as JSON for later pipeline stages."""
    logger.debug("target_engineering.saving_metadata", extra={"path": str(path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2)
    logger.info("target_engineering.metadata_saved", extra={"path": str(path)})


def load_transform_metadata(path: Path) -> TargetTransformMetadata:
    """Load previously persisted inverse-transform metadata."""
    logger.debug("target_engineering.loading_metadata", extra={"path": str(path)})
    if not path.exists():
        raise TargetEngineeringError(f"Transform metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        metadata = TargetTransformMetadata(**data)
        validate_target_metadata(metadata)
        return metadata
    except TypeError as exc:
        raise TargetEngineeringError(
            f"Malformed transform metadata at {path}: {exc}"
        ) from exc
