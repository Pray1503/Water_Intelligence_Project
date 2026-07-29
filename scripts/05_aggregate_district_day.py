"""
Stage 4.5: District-Day Aggregation
====================================

Pipeline runner for Stage 4.5. Reads every cleaned CSV under
data/cleaned, groups them by measurement category (consolidating
multi-year files such as the five yearly humidity datasets into one),
aggregates each category to (District LGD Code, Date), and writes:

  * data/aggregated/<category>_daily_district.csv
  * reports/aggregation/<category>_aggregation_report.json

Responsibilities of this script only: load config, discover cleaned
CSVs, group them by category, delegate all aggregation logic to
src.aggregation, save outputs, and print a final summary. All
reusable logic lives in src/aggregation.py.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import result

_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from src.aggregation import (
    CATEGORY_OUTPUT_FILENAMES,
    DEBUG,
    aggregate_to_district_day,
    check_no_duplicate_output_filenames,
    consolidate_category_files,
    detect_dataset_category,
    write_aggregation_report,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and return the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover_cleaned_csvs(cleaned_dir: Path) -> List[Path]:
    """Return every cleaned CSV under *cleaned_dir*, sorted for
    deterministic processing order.

    Raises
    ------
    FileNotFoundError
        If *cleaned_dir* contains no CSV files.
    """
    csv_files = sorted(cleaned_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No cleaned datasets found in {cleaned_dir}")
    return csv_files


def group_files_by_category(
    csv_files: List[Path],
) -> Dict[str, List[Path]]:
    """Group cleaned CSVs by detected measurement category.

    Multiple files detected as the same category (for example the five
    yearly humidity files) are grouped together so they can later be
    consolidated into a single dataset before aggregation.

    Raises
    ------
    CategoryDetectionError
        Propagated from detect_dataset_category if a file's category
        cannot be determined.
    """
    grouped: Dict[str, List[Path]] = {}
    for csv_path in csv_files:
        category = detect_dataset_category(csv_path.name)
        grouped.setdefault(category, []).append(csv_path)

        if DEBUG:
            print(f"[DETECT] {csv_path.name} -> category='{category}'")
        else:
            logger.info("Detected category '%s' for %s", category, csv_path.name)

    return grouped


def load_category_files(paths: List[Path]) -> List[Tuple[str, pd.DataFrame]]:
    """Load every CSV in *paths* into a (filename, dataframe) pair."""
    file_frames: List[Tuple[str, pd.DataFrame]] = []

    for path in paths:
        df = pd.read_csv(
            path,
            low_memory=False,
        )

        # Defensive cleanup in case CSV headers contain leading/trailing spaces.
        df.columns = [str(col).strip() for col in df.columns]

        if DEBUG:
            print(f"[LOAD] {path.name}")
            print(f"[ROWS] {len(df):,}")
            print(f"[COLUMNS] {len(df.columns)}")

        file_frames.append((path.name, df))

    return file_frames


def run_stage_4_5(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Execute Stage 4.5 end-to-end and return a summary dict."""
    paths = config.get("paths", {})
    cleaned_dir = project_root / paths.get("data_cleaned", "data/cleaned")
    aggregated_dir = project_root / paths.get("data_aggregated", "data/aggregated")
    reports_dir = project_root / paths.get("reports_dir", "reports")
    aggregation_reports_dir = reports_dir / "aggregation"

    aggregated_dir.mkdir(parents=True, exist_ok=True)
    aggregation_reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cleaned data directory   : %s", cleaned_dir)
    logger.info("Aggregated output dir    : %s", aggregated_dir)
    logger.info("Aggregation reports dir  : %s", aggregation_reports_dir)

    csv_files = discover_cleaned_csvs(cleaned_dir)
    logger.info("Found %d cleaned CSV(s) in %s", len(csv_files), cleaned_dir)

    grouped_files = group_files_by_category(csv_files)

    if DEBUG:
        print("\n" + "=" * 70)
        print("CATEGORY SUMMARY")
        print("=" * 70)

        for category, files in sorted(grouped_files.items()):
            print(f"{category:<15} -> {len(files)} file(s)")
            for file in files:
                print(f"    {file.name}")

        print("=" * 70)

    missing_categories = sorted(set(CATEGORY_OUTPUT_FILENAMES) - set(grouped_files))
    if missing_categories:
        raise FileNotFoundError(
            f"No cleaned CSV files were found for the following expected "
            f"categories: {missing_categories}. Stage 4.5 requires all of "
            f"{sorted(CATEGORY_OUTPUT_FILENAMES)} to be present."
        )

    output_filenames = [CATEGORY_OUTPUT_FILENAMES[c] for c in grouped_files]
    check_no_duplicate_output_filenames(output_filenames)

    summary_rows: List[Dict[str, Any]] = []

    for index, (category, category_paths) in enumerate(
        sorted(grouped_files.items()), start=1
    ):
        print(f"\n[{index}/{len(grouped_files)}] Processing category: {category}")

        file_frames = load_category_files(category_paths)

        if len(file_frames) > 1:
            if DEBUG:
                print(
                    f"[MERGE] category '{category}' has {len(file_frames)} "
                    "source files -- consolidating before aggregation"
                )
            consolidated_df = consolidate_category_files(file_frames)
            source_files = [name for name, _ in file_frames]
        else:
            consolidated_df = file_frames[0][1]
            source_files = [file_frames[0][0]]

        dataset_name = f"{category}_daily_district"
        aggregated_df, report = aggregate_to_district_day(
            df=consolidated_df,
            category=category,
            dataset_name=dataset_name,
            source_files=source_files,
        )

        output_filename = CATEGORY_OUTPUT_FILENAMES[category]
        output_path = aggregated_dir / output_filename
        aggregated_df.to_csv(output_path, index=False)

        if DEBUG:
            print(f"[SAVE] {category} -> {output_path}")
            print(f"[OUTPUT ROWS] {len(aggregated_df):,}")
        else:
            logger.info("Saved aggregated dataset: %s", output_path)

        report_path = aggregation_reports_dir / f"{category}_aggregation_report.json"
        write_aggregation_report(report, report_path)

        summary_rows.append(
            {
                "category": category,
                "output_file": str(output_path),
                "report_file": str(report_path),
                "input_rows": report["input_rows"],
                "output_rows": report["output_rows"],
                "district_count": report["district_count"],
            }
        )

    return {"summary_rows": summary_rows}


def print_final_summary(result: Dict[str, Any]) -> None:
    """Print the final Stage 4.5 summary table."""

    print("\n" + "=" * 70)
    print("STAGE 4.5 SUMMARY")
    print("=" * 70)

    for row in result["summary_rows"]:
        print(
            f"{row['category']:<15} "
            f"{row['input_rows']:>12,} -> {row['output_rows']:>8,} rows  "
            f"districts={row['district_count']:<4} "
            f"-> {row['output_file']}"
        )

    total_input = sum(row["input_rows"] for row in result["summary_rows"])

    total_output = sum(row["output_rows"] for row in result["summary_rows"])

    print("=" * 70)
    print(f"TOTAL INPUT ROWS : {total_input:,}")
    print(f"TOTAL OUTPUT ROWS: {total_output:,}")

    if total_output > 0:
        print(f"OVERALL COMPRESSION : " f"{total_input / total_output:.2f}:1")

    print("=" * 70)


def main() -> None:
    """Entry point -- resolve project root, load config, run Stage 4.5."""
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config" / "config.yaml"
    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info("Stage 4.5: District-Day Aggregation")
    logger.info("=" * 60)

    try:
        result = run_stage_4_5(config, root)

    except Exception:
        logger.exception("Stage 4.5 failed with an unexpected error.")
        raise

    print_final_summary(result)
    logger.info("Stage 4.5 complete.")


if __name__ == "__main__":
    main()
