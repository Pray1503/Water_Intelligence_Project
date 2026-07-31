"""Stage 10 - Layer 1: Dataset Loading.

Loads train.parquet, validation.parquet, and test.parquet (produced by
Stage 9) into an immutable DatasetBundle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.stage10.exceptions import DatasetLoadError
from src.stage10.logging_utils import log_call

logger = logging.getLogger("stage10")


@dataclass(frozen=True)
class DatasetBundle:
    """Immutable container for the three Stage 9 dataset splits.

    Attributes
    ----------
    train, validation, test:
        The loaded DataFrames for each split.
    train_path, validation_path, test_path:
        Source paths each split was loaded from (kept for traceability).
    loaded_at:
        UTC timestamp recorded at load time.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_path: Path
    validation_path: Path
    test_path: Path
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def splits(self) -> dict[str, pd.DataFrame]:
        """Return {"train": ..., "validation": ..., "test": ...}."""
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _validate_path_exists(path: Path, split_name: str) -> None:
    """Raise DatasetLoadError if *path* does not exist or is not a file."""
    if not path.exists():
        raise DatasetLoadError(f"{split_name} parquet file not found: {path}")
    if not path.is_file():
        raise DatasetLoadError(f"{split_name} path is not a file: {path}")


@log_call
def load_parquet_split(path: Path, split_name: str) -> pd.DataFrame:
    """Load a single parquet split file.

    Parameters
    ----------
    path:
        Path to the parquet file.
    split_name:
        Human-readable split name (e.g. "train"), used in error messages.

    Raises
    ------
    DatasetLoadError
        If the file is missing or cannot be parsed as parquet.
    """
    path = Path(path)
    _validate_path_exists(path, split_name)

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise DatasetLoadError(
            f"Failed to read {split_name} parquet file at {path}: {exc}"
        ) from exc

    logger.info(
        "Loaded %s split: %s rows, %s columns", split_name, len(df), len(df.columns)
    )
    return df


@log_call
def load_dataset_bundle(
    train_path: Path, validation_path: Path, test_path: Path
) -> DatasetBundle:
    """Load train/validation/test parquet files into an immutable
    DatasetBundle.

    Parameters
    ----------
    train_path, validation_path, test_path:
        Paths to the Stage 9 output parquet files.

    Returns
    -------
    DatasetBundle

    Raises
    ------
    DatasetLoadError
        If any split fails to load.
    """
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    test_path = Path(test_path)

    train_df = load_parquet_split(train_path, "train")
    validation_df = load_parquet_split(validation_path, "validation")
    test_df = load_parquet_split(test_path, "test")

    bundle = DatasetBundle(
        train=train_df,
        validation=validation_df,
        test=test_df,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
    )

    logger.info(
        "DatasetBundle assembled: train=%s validation=%s test=%s",
        len(bundle.train),
        len(bundle.validation),
        len(bundle.test),
    )
    return bundle
