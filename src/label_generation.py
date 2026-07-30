"""
Stage 8: Water Stress Label Generation.

Calculates the Water Stress Index (WSI) and future target variables (7, 15, and 30-day leads)
for the engineered features dataset.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Weights for WSI components
DEFAULT_WSI_WEIGHTS = {
    "groundwater": 0.40,
    "rainfall": 0.30,
    "temperature": 0.10,
    "humidity": 0.10,
    "river_level": 0.10,
}


def normalize_by_group(
    df: pd.DataFrame,
    column: str,
    group_col: str,
    invert: bool = False
) -> pd.Series:
    """
    Normalize a column within each group using min-max scaling.
    
    If invert is True, then:
        1.0 represents the minimum value (highest stress)
        0.0 represents the maximum value (lowest stress)
    """
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _min_max(x: pd.Series) -> pd.Series:
        if x.isna().all():
            return pd.Series(np.nan, index=x.index)
        xmin = x.min()
        xmax = x.max()
        if pd.isna(xmin) or pd.isna(xmax) or xmin == xmax:
            return pd.Series(0.0, index=x.index)
        
        normalized = (x - xmin) / (xmax - xmin)
        if invert:
            normalized = 1.0 - normalized
        return normalized

    return df.groupby(group_col, group_keys=False)[column].apply(_min_max)


def calculate_water_stress_index(
    df: pd.DataFrame,
    weights: Dict[str, float] = None,
    group_col: str = "District LGD Code",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate the composite Water Stress Index (WSI) based on normalized components.
    
    WSI ranges from 0.0 (no stress) to 1.0 (extreme water stress).
    Weights are dynamically re-scaled if any component is entirely missing for a row.
    """
    if weights is None:
        weights = DEFAULT_WSI_WEIGHTS

    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate stress components (0.0 = low stress, 1.0 = high stress)
    # 1. Groundwater: higher depth to water table = higher depletion/stress
    gw_stress = normalize_by_group(df, "groundwater_level", group_col, invert=False)

    # 2. Rainfall: lower 30-day rolling rainfall = higher stress
    rf_stress = normalize_by_group(df, "rainfall_mm_rolling_30d_sum", group_col, invert=True)

    # 3. Temperature: higher temperature = higher evapotranspiration/stress
    temp_stress = normalize_by_group(df, "air_temperature", group_col, invert=False)

    # 4. Humidity: lower humidity = higher evapotranspiration/stress
    rh_stress = normalize_by_group(df, "relative_humidity", group_col, invert=True)

    # 5. River level: lower river level = higher surface water stress
    river_stress = normalize_by_group(df, "river_level", group_col, invert=True)

    # Combine components in a DataFrame
    components_df = pd.DataFrame({
        "groundwater": gw_stress,
        "rainfall": rf_stress,
        "temperature": temp_stress,
        "humidity": rh_stress,
        "river_level": river_stress
    })

    # Vectorized weighted calculation with dynamic fallback for NaNs
    weighted_components = components_df.multiply(normalized_weights)
    weights_mask = components_df.notna().multiply(normalized_weights)
    sum_weights = weights_mask.sum(axis=1)

    # Avoid division by zero if all are NaN
    wsi = weighted_components.sum(axis=1) / sum_weights.replace(0, np.nan)
    wsi = wsi.fillna(0.5)  # Moderate default fallback
    wsi = wsi.clip(0.0, 1.0)

    # Add components back to df for logging/reporting
    components_out = components_df.copy()
    components_out["wsi"] = wsi
    
    return wsi, components_out


def generate_lead_targets(
    df: pd.DataFrame,
    target_col: str,
    leads: list[int],
    group_col: str = "District LGD Code",
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Generate target lead variables by shifting target_col forward in time per group.
    
    Leads are positive integers representing days in the future (e.g. 7, 15, 30).
    """
    df_out = df.copy()
    
    # Ensure sorted order
    df_out = df_out.sort_values(by=[group_col, date_col]).reset_index(drop=True)
    
    grouped = df_out.groupby(group_col)
    
    for lead in leads:
        lead_col = f"{target_col}_lead_{lead}"
        # A negative shift moves future values to the current row
        df_out[lead_col] = grouped[target_col].shift(-lead)
        
    return df_out


def generate_water_stress_labels(
    df: pd.DataFrame,
    weights: Dict[str, float] = None,
    leads: list[int] = None,
    group_col: str = "District LGD Code",
    date_col: str = "Date"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Orchestrate Stage 8: Generate Water Stress Index and future lead targets.
    """
    if leads is None:
        leads = [7, 15, 30]

    start_time = time.perf_counter()
    logger.info("Generating water stress labels and targets...")

    df_out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_out[date_col]):
        df_out[date_col] = pd.to_datetime(df_out[date_col])

    # 1. Calculate WSI
    wsi, components_df = calculate_water_stress_index(df_out, weights, group_col)
    df_out["wsi"] = wsi

    # Add component scores for transparency/XAI
    for col in components_df.columns:
        if col != "wsi":
            df_out[f"{col}_stress_score"] = components_df[col]

    # 2. Generate Lead Targets
    df_out = generate_lead_targets(df_out, "wsi", leads, group_col, date_col)

    elapsed_time = time.perf_counter() - start_time
    logger.info(
        "Water stress labels generated successfully in %.3f seconds. Shape: %s",
        elapsed_time,
        df_out.shape
    )

    # Compile report summary
    report = {
        "status": "SUCCESS",
        "total_records": len(df_out),
        "columns_added": ["wsi"] + [f"{col}_stress_score" for col in components_df.columns if col != "wsi"] + [f"wsi_lead_{l}" for l in leads],
        "wsi_statistics": {
            "mean": float(df_out["wsi"].mean()),
            "std": float(df_out["wsi"].std()),
            "min": float(df_out["wsi"].min()),
            "max": float(df_out["wsi"].max()),
        },
        "lead_target_null_counts": {
            f"wsi_lead_{lead}": int(df_out[f"wsi_lead_{lead}"].isna().sum())
            for lead in leads
        },
        "execution_time_seconds": elapsed_time
    }

    return df_out, report
