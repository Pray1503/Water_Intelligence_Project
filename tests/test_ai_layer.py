"""
Unit tests for the AI Layer.
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from src.label_generation import calculate_water_stress_index, generate_lead_targets, normalize_by_group
from src.simulation import run_intervention_simulation


def test_normalize_by_group_bounds():
    """Verify that group-wise normalization correctly bounds values between 0.0 and 1.0."""
    df = pd.DataFrame({
        "District LGD Code": [1, 1, 1, 2, 2, 2],
        "value": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
    })
    
    norm = normalize_by_group(df, "value", "District LGD Code", invert=False)
    assert norm.iloc[0] == 0.0
    assert norm.iloc[1] == 0.5
    assert norm.iloc[2] == 1.0
    assert norm.iloc[3] == 0.0
    assert norm.iloc[4] == 0.5
    assert norm.iloc[5] == 1.0


def test_normalize_by_group_inverted():
    """Verify that inverted group normalization maps maximum values to 0.0 and minimums to 1.0."""
    df = pd.DataFrame({
        "District LGD Code": [1, 1, 1],
        "value": [10.0, 20.0, 30.0]
    })
    
    norm = normalize_by_group(df, "value", "District LGD Code", invert=True)
    assert norm.iloc[0] == 1.0
    assert norm.iloc[1] == 0.5
    assert norm.iloc[2] == 0.0


def test_calculate_wsi_without_nans():
    """Verify WSI calculation with complete data."""
    df = pd.DataFrame({
        "District LGD Code": [1, 1, 1],
        "groundwater_level": [5.0, 10.0, 15.0],
        "rainfall_mm_rolling_30d_sum": [100.0, 50.0, 0.0],
        "air_temperature": [30.0, 35.0, 40.0],
        "relative_humidity": [80.0, 50.0, 20.0],
        "river_level": [5.0, 3.0, 1.0]
    })
    
    wsi, comp = calculate_water_stress_index(df, group_col="District LGD Code")
    
    # Values should grow from low stress to high stress
    assert wsi.iloc[0] == 0.0
    assert wsi.iloc[2] == 1.0
    assert (wsi >= 0.0).all() and (wsi <= 1.0).all()


def test_calculate_wsi_with_nans_reweights_dynamically():
    """Verify WSI calculation handles missing components by re-scaling weights dynamically."""
    df = pd.DataFrame({
        "District LGD Code": [1, 1, 1],
        "groundwater_level": [5.0, 10.0, 15.0],
        "rainfall_mm_rolling_30d_sum": [100.0, 50.0, 0.0],
        "air_temperature": [30.0, 35.0, 40.0],
        "relative_humidity": [np.nan, np.nan, np.nan],  # Entirely missing
        "river_level": [np.nan, np.nan, np.nan]          # Entirely missing
    })
    
    wsi, comp = calculate_water_stress_index(df, group_col="District LGD Code")
    
    # Should still compute successfully using non-missing weights
    assert wsi.iloc[0] == 0.0
    assert wsi.iloc[2] == 1.0
    assert (wsi >= 0.0).all() and (wsi <= 1.0).all()


def test_generate_lead_targets():
    """Verify lead shifting puts future target values on the current date, grouped by district."""
    df = pd.DataFrame({
        "District LGD Code": [1, 1, 1, 2, 2, 2],
        "Date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01", "2025-01-02", "2025-01-03"]),
        "wsi": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
    })
    
    shifted = generate_lead_targets(df, "wsi", leads=[1, 2], group_col="District LGD Code", date_col="Date")
    
    # For District 1:
    assert shifted.loc[0, "wsi_lead_1"] == 0.2
    assert shifted.loc[0, "wsi_lead_2"] == 0.3
    assert pd.isna(shifted.loc[1, "wsi_lead_2"])
    
    # For District 2:
    assert shifted.loc[3, "wsi_lead_1"] == 0.7
    assert shifted.loc[3, "wsi_lead_2"] == 0.8
    assert pd.isna(shifted.loc[4, "wsi_lead_2"])


def test_intervention_simulation_changes():
    """Verify that policy interventions correctly decrease groundwater depth and increase river level."""
    df = pd.DataFrame({
        "groundwater_level": [10.0],
        "river_level": [2.0],
        "rainfall_mm": [0.0],
        # Lags
        "groundwater_level_lag_1": [10.0],
        "river_level_lag_1": [2.0],
        "rainfall_mm_lag_1": [0.0],
        # Rolling
        "groundwater_level_rolling_30d_mean": [10.0],
        "river_level_rolling_7d_mean": [2.0],
        "rainfall_mm_rolling_30d_sum": [0.0]
    })
    
    # Apply rainwater harvesting at 100% intensity
    simulated = run_intervention_simulation(
        df, rainwater_harvesting=1.0, demand_reduction=0.0, water_conservation=0.0
    )
    
    # Groundwater table rises (value decreases)
    assert simulated.loc[0, "groundwater_level"] == 7.0
    assert simulated.loc[0, "groundwater_level_lag_1"] == 7.0
    assert simulated.loc[0, "groundwater_level_rolling_30d_mean"] == 7.0
    
    # River level rises
    assert simulated.loc[0, "river_level"] == 2.8
    assert simulated.loc[0, "river_level_lag_1"] == 2.8
    assert simulated.loc[0, "river_level_rolling_7d_mean"] == 2.8
