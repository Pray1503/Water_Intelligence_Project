import logging
import sys
from pathlib import Path
import json

script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != script_dir]

import yaml
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def extract_schema(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract schema dictionary mapping column name to data type
    and preserve column order.
    """
    cols = profile_data.get("Profile", {}).get("Column Profiles", [])

    schema = {}
    ordered_names = []

    for col in cols:
        name = col.get("Name")
        dtype = col.get("Data Type")

        if name:
            schema[name] = dtype
            ordered_names.append(name)

    return {
        "types": schema,
        "order": ordered_names
    }


def validate_dataset(
    baseline_schema: Dict[str, Any],
    dataset_schema: Dict[str, Any]
) -> Dict[str, Any]:

    errors = []
    warnings = []

    base_types = baseline_schema["types"]
    base_order = baseline_schema["order"]

    ds_types = dataset_schema["types"]
    ds_order = dataset_schema["order"]

    # Numeric types that are considered compatible
    numeric_types = {
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64"
    }

    # -------------------------------
    # Missing Columns
    # -------------------------------
    for col in base_types:
        if col not in ds_types:
            errors.append(
                f"Missing required column: '{col}'"
            )

    # -------------------------------
    # Unexpected Columns
    # -------------------------------
    for col in ds_types:
        if col not in base_types:
            errors.append(
                f"Unexpected column found: '{col}'"
            )

    # -------------------------------
    # Data Type Validation
    # -------------------------------
    common_cols = set(base_types.keys()).intersection(
        ds_types.keys()
    )

    for col in common_cols:

        base_type = str(base_types[col]).strip().lower()
        current_type = str(ds_types[col]).strip().lower()
        print(f"{col} -> baseline={base_type}, current={current_type}")
        # Exact Match
        if base_type == current_type:
          continue

        # Compatible numeric types
        if base_type in numeric_types and current_type in numeric_types:
            warnings.append(
                f"Compatible numeric type difference for '{col}': "
                f"Baseline={base_type}, Current={current_type}"
            )
            continue

        # Actual mismatch
        errors.append(
            f"Data type mismatch for '{col}': "
            f"Expected {base_type}, Found {current_type}"
        )

    # -------------------------------
    # Column Order
    # -------------------------------
    base_common_order = [
        c for c in base_order
        if c in common_cols
    ]

    ds_common_order = [
        c for c in ds_order
        if c in common_cols
    ]

    if base_common_order != ds_common_order:
        warnings.append(
            "Column order differs from baseline schema."
        )

    # -------------------------------
    # Final Status
    # -------------------------------
    if errors:
        status = "FAIL"

    elif warnings:
        status = "PASS WITH WARNINGS"

    else:
        status = "PASS"

    return {
        "Validation Status": status,
        "Error Count": len(errors),
        "Warning Count": len(warnings),
        "Errors": errors,
        "Warnings": warnings
    }
    
def validate_schemas(config: Dict[str, Any], project_root: Path) -> None:
    paths = config.get("paths", {})

    profiling_dir = project_root / paths.get(
        "profiling_dir",
        "reports/profiling"
    )

    validation_dir = project_root / paths.get(
        "validation_dir",
        "reports/validation"
    )

    if not profiling_dir.exists():
        logger.error(
            f"Profiling directory not found at {profiling_dir}"
        )
        sys.exit(1)

    validation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------
    # Load all profiling reports
    # -----------------------------------
    profiles_by_category = {}

    profile_files = list(
        profiling_dir.glob("*.json")
    )

    if not profile_files:
        logger.error(
            "No profiling JSON files found."
        )
        sys.exit(1)

    logger.info(
        "Loading profiles for schema validation..."
    )

    for pf in profile_files:

        try:
            with open(pf, "r") as f:
                data = json.load(f)

            category = data.get(
                "Metadata",
                {}
            ).get(
                "Dataset Category",
                "Unknown"
            )

            profiles_by_category.setdefault(
                category,
                []
            ).append(
                (pf.stem, data)
            )

        except Exception as e:
            logger.error(
                f"Failed to read {pf}: {e}"
            )

    # -----------------------------------
    # Summary Report
    # -----------------------------------
    summary_report = {
        "Overall Status": "PASS",
        "Total Datasets": len(profile_files),
        "Total Errors": 0,
        "Total Warnings": 0,
        "Dataset Summaries": {}
    }

    logger.info("Validating schemas...")

    # -----------------------------------
    # Validate each category
    # -----------------------------------
    for category, datasets in profiles_by_category.items():

        datasets.sort(
            key=lambda x: x[0]
        )

        baseline_name, baseline_data = datasets[0]

        logger.info(
            f"Category: {category} | "
            f"Baseline: {baseline_name}"
        )

        baseline_schema = extract_schema(
            baseline_data
        )

        for ds_name, ds_data in datasets:

            ds_schema = extract_schema(
                ds_data
            )

            validation_result = validate_dataset(
                baseline_schema,
                ds_schema
            )

            report_data = {
                "Dataset Name": ds_name,
                "Dataset Category": category,
                "Baseline Used": baseline_name,
                "Validation Results": validation_result
            }

            out_file = (
                validation_dir /
                f"{ds_name}_validation.json"
            )

            try:
                with open(out_file, "w") as f:
                    json.dump(
                        report_data,
                        f,
                        indent=4
                    )

            except Exception as e:
                logger.error(
                    f"Failed to write validation report "
                    f"for {ds_name}: {e}"
                )

            summary_report["Dataset Summaries"][ds_name] = {
                "Category": category,
                "Status": validation_result["Validation Status"],
                "Error Count": validation_result["Error Count"],
                "Warning Count": validation_result["Warning Count"]
            }

            summary_report["Total Errors"] += (
                validation_result["Error Count"]
            )

            summary_report["Total Warnings"] += (
                validation_result["Warning Count"]
            )

            logger.info(
                f"  [{validation_result['Validation Status']}] "
                f"{ds_name}"
            )

    # -----------------------------------
    # Overall Status
    # -----------------------------------
    if summary_report["Total Errors"] > 0:
        summary_report["Overall Status"] = "FAIL"

    elif summary_report["Total Warnings"] > 0:
        summary_report["Overall Status"] = "PASS WITH WARNINGS"

    else:
        summary_report["Overall Status"] = "PASS"

    summary_path = (
        validation_dir /
        "validation_summary.json"
    )

    try:
        with open(summary_path, "w") as f:
            json.dump(
                summary_report,
                f,
                indent=4
            )

        logger.info(
            "Saved validation summary to "
            f"{summary_path.relative_to(project_root).as_posix()}"
        )

    except Exception as e:
        logger.error(
            f"Failed to write validation summary: {e}"
        )

    logger.info(
        "Schema validation completed."
    )


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    config_path = (
        project_root /
        "config" /
        "config.yaml"
    )

    config = load_config(
        config_path
    )

    validate_schemas(
        config,
        project_root
    )


if __name__ == "__main__":
    main()