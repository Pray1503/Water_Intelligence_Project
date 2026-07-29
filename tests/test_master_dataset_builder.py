import pandas as pd
import pytest

from src.merger import MergeExplosionError, MergeKeyError, merge_datasets


def _std(timestamps, station=None, district=None, **measurement_cols):
    """Build a minimal already-standardized frame like standardize_dataset
    would produce (timestamp always present; station/district optional)."""
    data = {"timestamp": pd.to_datetime(timestamps)}
    if station is not None:
        data["station"] = pd.array(station, dtype="string")
    if district is not None:
        data["district"] = pd.array(district, dtype="string")
    data.update(measurement_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Problem 1: many-to-many merge explosions must be prevented, not tolerated.
# ---------------------------------------------------------------------------


def test_duplicate_keys_on_both_sides_raises_instead_of_exploding():
    """Two rows sharing a key on BOTH sides must never silently produce a
    4-row cartesian blow-up. This must raise, not merge."""
    left = _std(
        ["2024-01-01", "2024-01-01"],
        station=["A", "A"],
        value=[1.0, 2.0],
    )
    right = _std(
        ["2024-01-01", "2024-01-01"],
        station=["A", "A"],
        other=[10.0, 20.0],
    )

    with pytest.raises(MergeExplosionError):
        merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])


def test_duplicate_keys_on_one_side_only_is_a_legitimate_fan_out():
    """Many stations sharing a district+timestamp (right side unique) is a
    normal one-to-many join, not an explosion, and must be allowed."""
    left = _std(
        ["2024-01-01", "2024-01-01"],
        district=["Ahmedabad", "Ahmedabad"],
        station=["A", "B"],
        value=[1.0, 2.0],
    )
    right = _std(
        ["2024-01-01"],
        district=["Ahmedabad"],
        station=[pd.NA],
        other=[100.0],
    )
    # Station is unusable on the right (all null) so this must fall back to
    # timestamp+district, where the right side has a single unique key and
    # the left has two rows sharing it -- a valid many-to-one fan-out.
    merged, summary = merge_datasets(
        [left, right], dataset_names=["left_ds", "right_ds"]
    )

    assert len(merged) == 2
    assert summary["steps"][0]["merge_strategy"] == "timestamp_district"
    assert (merged["other"] == 100.0).all()


# ---------------------------------------------------------------------------
# Problem 5 / architectural rule 2: never merge on timestamp alone.
# ---------------------------------------------------------------------------


def test_raises_when_neither_station_nor_district_is_usable():
    left = _std(["2024-01-01"], value=[1.0])
    right = _std(["2024-01-01"], other=[2.0])

    with pytest.raises(MergeKeyError):
        merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])


def test_raises_when_station_and_district_are_entirely_null():
    left = _std(["2024-01-01"], station=[pd.NA], district=[pd.NA], value=[1.0])
    right = _std(["2024-01-01"], station=[pd.NA], district=[pd.NA], other=[2.0])

    with pytest.raises(MergeKeyError):
        merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])


def test_raises_when_timestamp_is_entirely_null():
    left = _std([pd.NaT], station=["A"], value=[1.0])
    right = _std(["2024-01-01"], station=["A"], other=[2.0])

    with pytest.raises(MergeKeyError):
        merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])


# ---------------------------------------------------------------------------
# Merge key priority: station preferred over district when both usable.
# ---------------------------------------------------------------------------


def test_prefers_station_over_district_when_both_usable():
    left = _std(["2024-01-01"], station=["A"], district=["Ahmedabad"], value=[1.0])
    right = _std(["2024-01-01"], station=["A"], district=["Ahmedabad"], other=[2.0])

    merged, summary = merge_datasets(
        [left, right], dataset_names=["left_ds", "right_ds"]
    )

    assert summary["steps"][0]["merge_keys"] == ["timestamp", "station"]
    assert summary["steps"][0]["merge_strategy"] == "timestamp_station"


def test_falls_back_to_district_when_station_unusable_on_either_side():
    left = _std(["2024-01-01"], station=["A"], district=["Ahmedabad"], value=[1.0])
    right = _std(["2024-01-01"], station=[pd.NA], district=["Ahmedabad"], other=[2.0])

    merged, summary = merge_datasets(
        [left, right], dataset_names=["left_ds", "right_ds"]
    )

    assert summary["steps"][0]["merge_keys"] == ["timestamp", "district"]
    assert summary["steps"][0]["merge_strategy"] == "timestamp_district"


# ---------------------------------------------------------------------------
# No data loss: outer join must preserve rows unique to either side.
# ---------------------------------------------------------------------------


def test_outer_join_preserves_rows_unique_to_either_dataset():
    left = _std(["2024-01-01"], station=["A"], value=[1.0])
    right = _std(["2024-01-02"], station=["B"], other=[2.0])

    merged, summary = merge_datasets(
        [left, right], dataset_names=["left_ds", "right_ds"]
    )

    assert len(merged) == 2
    assert summary["steps"][0]["left_only_rows"] == 1
    assert summary["steps"][0]["right_only_rows"] == 1
    assert summary["steps"][0]["matched_rows"] == 0


# ---------------------------------------------------------------------------
# Measurement columns must be preserved exactly, never aggregated/altered.
# ---------------------------------------------------------------------------


def test_measurement_values_are_preserved_exactly_not_aggregated():
    left = _std(["2024-01-01"], station=["A"], rainfall_mm=[12.5])
    right = _std(["2024-01-01"], station=["A"], humidity_pct=[88.0])

    merged, _ = merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])

    assert merged.loc[0, "rainfall_mm"] == 12.5
    assert merged.loc[0, "humidity_pct"] == 88.0


def test_overlapping_column_names_are_suffixed_by_dataset_not_overwritten():
    left = _std(["2024-01-01"], station=["A"], Agency=["AgencyOne"], value=[1.0])
    right = _std(["2024-01-01"], station=["A"], Agency=["AgencyTwo"], other=[2.0])

    merged, _ = merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])

    assert merged.loc[0, "Agency"] == "AgencyOne"
    assert merged.loc[0, "Agency__right_ds"] == "AgencyTwo"


def test_shared_descriptor_columns_are_coalesced_not_duplicated():
    """district/latitude/longitude describe the same reading on both sides
    of a station-keyed merge and must end up as a single column each --
    this is the 'standardize district/station/latitude/longitude' step,
    not a measurement, so it must not be left duplicated per dataset."""
    left = _std(
        ["2024-01-01"],
        station=["A"],
        district=["Ahmedabad"],
        value=[1.0],
    )
    left["latitude"] = [23.03]
    left["longitude"] = [72.58]
    right = _std(
        ["2024-01-01"],
        station=["A"],
        district=["Ahmedabad"],
        other=[2.0],
    )
    right["latitude"] = [23.03]
    right["longitude"] = [72.58]

    merged, _ = merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])

    assert "district__right_ds" not in merged.columns
    assert "latitude__right_ds" not in merged.columns
    assert "longitude__right_ds" not in merged.columns
    assert merged.loc[0, "district"] == "Ahmedabad"
    assert merged.loc[0, "latitude"] == 23.03


def test_shared_descriptor_column_falls_back_to_right_when_left_is_null():
    left = _std(["2024-01-01"], station=["A"], district=[pd.NA], value=[1.0])
    right = _std(["2024-01-01"], station=["A"], district=["Ahmedabad"], other=[2.0])

    merged, _ = merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])

    assert merged.loc[0, "district"] == "Ahmedabad"


# ---------------------------------------------------------------------------
# Merge statistics completeness (Problem 2).
# ---------------------------------------------------------------------------


def test_merge_summary_contains_full_step_diagnostics():
    left = _std(["2024-01-01"], station=["A"], value=[1.0])
    right = _std(["2024-01-01"], station=["A"], other=[2.0])

    _, summary = merge_datasets([left, right], dataset_names=["left_ds", "right_ds"])

    assert summary["datasets_merged"] == ["left_ds", "right_ds"]
    assert summary["final_rows"] == 1
    step = summary["steps"][0]
    for key in (
        "left_dataset",
        "right_dataset",
        "merge_keys",
        "merge_strategy",
        "validate_mode",
        "left_rows_before",
        "right_rows_before",
        "rows_after",
        "matched_rows",
        "left_only_rows",
        "right_only_rows",
        "left_duplicate_key_groups",
        "right_duplicate_key_groups",
    ):
        assert key in step


def test_empty_dataset_list_returns_empty_summary():
    merged, summary = merge_datasets([])
    assert merged.empty
    assert summary == {
        "datasets_merged": [],
        "steps": [],
        "final_rows": 0,
        "final_columns": 0,
    }


def test_single_dataset_passthrough_requires_no_merge_keys():
    only = _std(["2024-01-01"], value=[1.0])  # no station/district at all

    merged, summary = merge_datasets([only], dataset_names=["only_ds"])

    assert len(merged) == 1
    assert summary["steps"] == []
