"""
Stage 6 - Build Master Dataset

Driver script for the Stage 6 Master Dataset Builder.

Responsibilities
----------------
- Load project configuration.
- Resolve project directories.
- Configure logging.
- Execute the Stage 6 pipeline.
- Print execution summary.

All business logic lives in:

    src/master_dataset.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.master_dataset import build_master_dataset

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

SCRIPT_NAME = "06_build_master_dataset"


def load_config() -> dict:
    """
    Load config.yaml.
    """

    config_path = PROJECT_ROOT / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:

    start = time.perf_counter()

    print("=" * 70)
    print("WATER INTELLIGENCE PLATFORM")
    print("STAGE 6 - MASTER DATASET")
    print("=" * 70)

    config = load_config()

    data_config = config["paths"]

    aggregated_directory = PROJECT_ROOT / data_config["data_aggregated"]

    processed_directory = PROJECT_ROOT / data_config["data_processed"]

    reports_directory = PROJECT_ROOT / data_config["master_dataset_reports"]

    master_df = build_master_dataset(
        aggregated_directory=aggregated_directory,
        processed_directory=processed_directory,
        reports_directory=reports_directory,
    )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("STAGE 6 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Rows      : {len(master_df):,}")
    print(f"Columns   : {len(master_df.columns)}")
    print(f"Time      : {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        logger.exception("%s failed.", SCRIPT_NAME)

        print()
        print("=" * 70)
        print("STAGE 6 FAILED")
        print("=" * 70)
        print(exc)

        raise
