"""
Stage 6 merge engine.

Combines standardized per-dataset frames into a single master frame while
enforcing the architectural rules for Stage 6:

  * Never merge using timestamp alone (timestamp + station, else
    timestamp + district, else raise).
  * Never allow a many-to-many merge to silently explode row counts.
  * Never aggregate, engineer, or normalize measurement values.
  * Preserve every measurement column exactly as it was standardized.

No spatial/GIS logic, no feature engineering, and no ML lives here by
design -- see project instructions for Stage 6 scope.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Columns considered part of the shared "key" schema produced by
# src.loaders.standardize_dataset. Used only to decide which columns are
# eligible to become merge keys -- never to drop or alter data.
_TIMESTAMP_COL = "timestamp"
_STATION_COL = "station"
_DISTRICT_COL = "district"
_LATITUDE_COL = "latitude"
_LONGITUDE_COL = "longitude"

# The standardized shared-key columns produced by standardize_dataset for
# every dataset. When two of these collide during a merge they describe the
# same real-world attribute (the district/station/coordinates of a single
# reading) rather than two different measurements, so they are coalesced
# into one column instead of being suffixed into duplicates. This is what
# Stage 6 step 2, "standardize timestamp/district/station/latitude/
# longitude," requires -- it is column unification, not aggregation:
# exactly one row goes in and exactly one row comes out, and no
# measurement column is touched by it.
_SHARED_DESCRIPTOR_COLS = [_DISTRICT_COL, _STATION_COL, _LATITUDE_COL, _LONGITUDE_COL]


class MergeKeyError(Exception):
    """Raised when no safe merge key combination exists between two datasets.

    Per architectural rule #2, timestamp-only merging is never permitted.
    If neither station nor district can be used, Stage 6 must stop rather
    than guess.
    """


class MergeExplosionError(Exception):
    """Raised when both sides of a merge have duplicate merge keys.

    A duplicate key on one side alone is a legitimate one-to-many fan-out
    (e.g. many stations sharing a district+timestamp key). Duplicates on
    BOTH sides simultaneously would multiply rows in a way that fabricates
    measurement combinations that never existed in the source data --
    that is the "many-to-many merge explosion" this project forbids.
    """


def _is_usable(df: pd.DataFrame, column: str) -> bool:
    """A column is usable as a merge key component if it exists and has
    at least one non-null value."""
    return column in df.columns and df[column].notna().any()


def _duplicate_key_stats(df: pd.DataFrame, keys: List[str]) -> Dict[str, int]:
    """Describe how many merge-key groups in *df* contain duplicates."""
    if df.empty:
        return {"duplicate_key_groups": 0, "max_group_size": 0, "duplicate_rows": 0}
    group_sizes = df.groupby(keys, dropna=False).size()
    duplicated = group_sizes[group_sizes > 1]
    return {
        "duplicate_key_groups": int(duplicated.shape[0]),
        "max_group_size": int(group_sizes.max()) if not group_sizes.empty else 0,
        "duplicate_rows": int(duplicated.sum()) if not duplicated.empty else 0,
    }


def _select_merge_keys(
    left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str
) -> Tuple[List[str], str]:
    """Choose merge keys under the mandated priority order, or raise.

    Priority: timestamp + station -> timestamp + district -> raise.
    Timestamp alone is never an acceptable fallback.
    """
    if not _is_usable(left, _TIMESTAMP_COL) or not _is_usable(right, _TIMESTAMP_COL):
        raise MergeKeyError(
            f"Cannot merge '{right_name}' into '{left_name}': timestamp is "
            "missing or entirely null in one of the datasets, so no merge "
            "key can be formed. Timestamp-only merging is not permitted, "
            "so this is a hard stop rather than a fallback."
        )

    if _is_usable(left, _STATION_COL) and _is_usable(right, _STATION_COL):
        return [_TIMESTAMP_COL, _STATION_COL], "timestamp_station"

    if _is_usable(left, _DISTRICT_COL) and _is_usable(right, _DISTRICT_COL):
        return [_TIMESTAMP_COL, _DISTRICT_COL], "timestamp_district"

    raise MergeKeyError(
        f"Cannot merge '{right_name}' into '{left_name}': station is "
        "unavailable in at least one dataset and district is also "
        "unavailable in at least one dataset. Merging on timestamp alone "
        "is forbidden, so this merge cannot proceed safely."
    )


def _merge_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Merge two standardized frames on the mandated key priority.

    Uses an outer join so that no measurement row from either dataset is
    silently dropped -- Stage 6 is building a comprehensive statewide
    master table from sensor networks that only partially overlap by
    station, so a left/inner join would quietly discard entire datasets'
    worth of readings whenever their stations don't match the anchor
    dataset. Rows that don't share a key simply carry nulls for the
    other dataset's measurement columns, which is the documented,
    non-aggregating, non-imputing behavior Stage 6 must have.
    """
    merge_keys, strategy = _select_merge_keys(left, right, left_name, right_name)

    left_dup = _duplicate_key_stats(left, merge_keys)
    right_dup = _duplicate_key_stats(right, merge_keys)

    if left_dup["duplicate_key_groups"] > 0 and right_dup["duplicate_key_groups"] > 0:
        raise MergeExplosionError(
            f"Refusing to merge '{right_name}' into '{left_name}' on "
            f"{merge_keys}: both sides contain duplicate merge keys "
            f"({left_dup['duplicate_key_groups']} duplicated key group(s) "
            f"in '{left_name}', {right_dup['duplicate_key_groups']} in "
            f"'{right_name}'). Proceeding would create a many-to-many "
            "merge that fabricates measurement combinations not present "
            "in the source data. Stage 6 does not aggregate measurements, "
            "so this must be resolved upstream (e.g. in Stage 4/5) before "
            "these two datasets can be merged."
        )

    if left_dup["duplicate_key_groups"] > 0:
        validate = "m:1"
    elif right_dup["duplicate_key_groups"] > 0:
        validate = "1:m"
    else:
        validate = "1:1"

    right_suffix = f"__{right_name}"
    try:
        merged = pd.merge(
            left,
            right,
            on=merge_keys,
            how="outer",
            suffixes=("", right_suffix),
            validate=validate,
            indicator=True,
        )
    except pd.errors.MergeError as exc:
        # Defensive backstop: our pre-check above should already have
        # caught any many-to-many condition, but if pandas detects a
        # violation we didn't anticipate, fail loudly rather than let
        # rows silently multiply.
        raise MergeExplosionError(
            f"Merging '{right_name}' into '{left_name}' on {merge_keys} "
            f"violated the expected '{validate}' cardinality: {exc}"
        ) from exc

    # Coalesce the shared standardized descriptor columns that collided but
    # weren't used as this step's merge key (the key columns are already
    # unified by pandas via `on=`). Left's value wins when both are
    # present; right only fills in where left is null. Measurement columns
    # are never touched by this -- only district/station/latitude/
    # longitude.
    for col in _SHARED_DESCRIPTOR_COLS:
        right_col = f"{col}{right_suffix}"
        if right_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[right_col])
            else:
                merged[col] = merged[right_col]
            merged = merged.drop(columns=[right_col])

    match_counts = merged["_merge"].value_counts()
    step_stats = {
        "left_dataset": left_name,
        "right_dataset": right_name,
        "merge_keys": merge_keys,
        "merge_strategy": strategy,
        "validate_mode": validate,
        "left_rows_before": int(len(left)),
        "right_rows_before": int(len(right)),
        "rows_after": int(len(merged)),
        "matched_rows": int(match_counts.get("both", 0)),
        "left_only_rows": int(match_counts.get("left_only", 0)),
        "right_only_rows": int(match_counts.get("right_only", 0)),
        "left_duplicate_key_groups": left_dup["duplicate_key_groups"],
        "right_duplicate_key_groups": right_dup["duplicate_key_groups"],
    }

    merged = merged.drop(columns=["_merge"])
    return merged, step_stats


def merge_datasets(
    datasets: List[pd.DataFrame],
    dataset_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Merge standardized datasets under Stage 6's mandated rules.

    Sequentially outer-merges each dataset into a running master frame.
    At every step the merge key is chosen by priority (timestamp+station,
    then timestamp+district, else raise) and a many-to-many explosion is
    detected and rejected before pandas ever executes the join.

    Parameters
    ----------
    datasets:
        Standardized frames (output of src.loaders.standardize_dataset),
        in the order they should be merged.
    dataset_names:
        Optional human-readable names, same length as *datasets*, used to
        make error messages and merge statistics traceable to a source
        file. Defaults to "dataset_0", "dataset_1", ... when omitted.

    Returns
    -------
    (merged_df, merge_summary) where merge_summary contains a per-step
    breakdown plus overall totals.
    """
    if not datasets:
        return pd.DataFrame(), {
            "datasets_merged": [],
            "steps": [],
            "final_rows": 0,
            "final_columns": 0,
        }

    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
    if len(dataset_names) != len(datasets):
        raise ValueError("dataset_names must be the same length as datasets")

    merged = datasets[0].copy()
    current_name = dataset_names[0]
    steps: List[Dict[str, Any]] = []

    for df, name in zip(datasets[1:], dataset_names[1:]):
        merged, step_stats = _merge_pair(merged, df, current_name, name)
        steps.append(step_stats)
        # Every step after the first represents the accumulated master
        # frame so far; name it accordingly for the next iteration's
        # error messages and stats.
        current_name = f"({current_name}+{name})"

    merge_summary: Dict[str, Any] = {
        "datasets_merged": dataset_names,
        "steps": steps,
        "final_rows": int(len(merged)),
        "final_columns": int(len(merged.columns)),
    }
    return merged, merge_summary
