"""
Stage 10: Simulation and Recommendation Engine.

Provides the logic to simulate policy interventions (rainwater harvesting, demand reduction,
water conservation) and generate actionable recommendations based on predicted stress drivers.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def run_intervention_simulation(
    df: pd.DataFrame,
    rainwater_harvesting: float,     # 0.0 to 1.0
    demand_reduction: float,         # 0.0 to 1.0
    water_conservation: float,       # 0.0 to 1.0
    additional_water_supply: float = 0.0,  # 0.0 to 1.0
) -> pd.DataFrame:
    """
    Simulate the physical impact of policy interventions by modifying features and
    propagating updates to lag and rolling features.
    
    Returns a copy of the dataframe with simulated features.
    """
    sim_df = df.copy()

    # 1. Calculate delta values based on input intensities
    # Rainwater Harvesting: recharges groundwater and increases river levels
    gw_recharge_rwh = 3.0 * rainwater_harvesting
    river_boost_rwh = 0.8 * rainwater_harvesting
    rain_boost_rwh = 10.0 * rainwater_harvesting  # simulated effective rain (mm)

    # Demand Reduction: lowers extraction (improves groundwater levels)
    gw_recharge_dr = 2.0 * demand_reduction

    # Water Conservation: improves groundwater table and river levels
    gw_recharge_wc = 1.5 * water_conservation
    river_boost_wc = 0.5 * water_conservation

    # Additional Water Supply: directly augments surface flows and reduces groundwater pressure
    gw_recharge_aws = 1.0 * additional_water_supply
    river_boost_aws = 2.0 * additional_water_supply

    # Sum up the deltas
    total_gw_improvement = gw_recharge_rwh + gw_recharge_dr + gw_recharge_wc + gw_recharge_aws
    total_river_boost = river_boost_rwh + river_boost_wc + river_boost_aws
    total_rain_boost = rain_boost_rwh

    # 2. Propagate Groundwater updates (Note: lower depth value = higher water table = less stress)
    if "groundwater_level" in sim_df.columns:
        sim_df["groundwater_level"] = (sim_df["groundwater_level"] - total_gw_improvement).clip(lower=0.0)
        
        # Propagate to lags
        for lag in [1, 7, 30]:
            col = f"groundwater_level_lag_{lag}"
            if col in sim_df.columns:
                sim_df[col] = (sim_df[col] - total_gw_improvement).clip(lower=0.0)
                
        # Propagate to rolling means
        for win in [7, 30]:
            col = f"groundwater_level_rolling_{win}d_mean"
            if col in sim_df.columns:
                sim_df[col] = (sim_df[col] - total_gw_improvement).clip(lower=0.0)

    # 3. Propagate River Level updates (higher level = less stress)
    if "river_level" in sim_df.columns:
        sim_df["river_level"] = sim_df["river_level"] + total_river_boost
        
        # Propagate to lag
        if "river_level_lag_1" in sim_df.columns:
            sim_df["river_level_lag_1"] = sim_df["river_level_lag_1"] + total_river_boost
            
        # Propagate to rolling mean
        if "river_level_rolling_7d_mean" in sim_df.columns:
            sim_df["river_level_rolling_7d_mean"] = sim_df["river_level_rolling_7d_mean"] + total_river_boost

    # 4. Propagate Rainfall updates (higher sum/mean = less stress)
    if "rainfall_mm" in sim_df.columns:
        sim_df["rainfall_mm"] = sim_df["rainfall_mm"] + total_rain_boost
        
        # Propagate to lags
        for lag in [1, 7]:
            col = f"rainfall_mm_lag_{lag}"
            if col in sim_df.columns:
                sim_df[col] = sim_df[col] + total_rain_boost
                
        # Propagate to rolling means/sums
        for win in [7, 30]:
            mean_col = f"rainfall_mm_rolling_{win}d_mean"
            sum_col = f"rainfall_mm_rolling_{win}d_sum"
            if mean_col in sim_df.columns:
                sim_df[mean_col] = sim_df[mean_col] + total_rain_boost
            if sum_col in sim_df.columns:
                sim_df[sum_col] = sim_df[sum_col] + (total_rain_boost * win)

    return sim_df


def predict_water_stress_scenarios(
    df: pd.DataFrame,
    models: Dict[str, Any],
    rainwater_harvesting: float = 0.0,
    demand_reduction: float = 0.0,
    water_conservation: float = 0.0,
    additional_water_supply: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Run simulation and predict 7, 15, and 30-day WSI values.
    
    Includes a direct behavioral offset adjustment representing conservation effects
    not fully captured by physical sensors.
    """
    # 1. Run physical feature simulation
    sim_df = run_intervention_simulation(
        df, rainwater_harvesting, demand_reduction, water_conservation, additional_water_supply
    )

    # 2. Get predictions from models
    predictions = {}
    
    # Identify feature columns (match training inputs by dropping metadata)
    exclude_cols = ["Date", "District", "wsi_lead_7", "wsi_lead_15", "wsi_lead_30"]
    X_sim = sim_df.drop(columns=exclude_cols, errors="ignore")
    for col in X_sim.select_dtypes(include=["object", "string", "category"]).columns:
        X_sim[col] = X_sim[col].astype("category").cat.codes
    X_sim = X_sim.fillna(X_sim.median(numeric_only=True))

    # Calculate direct behavioral offset (represents efficiency and conservation policy impact)
    direct_offset = (
        (0.05 * demand_reduction)
        + (0.04 * water_conservation)
        + (0.03 * rainwater_harvesting)
        + (0.05 * additional_water_supply)
    )

    for target_col, model in models.items():
        # Get raw ML prediction
        preds = model.predict(X_sim)
        
        # Apply direct offset and clip to valid WSI range
        adjusted_preds = (preds - direct_offset).clip(0.0, 1.0)
        predictions[target_col] = adjusted_preds

    return predictions


def generate_recommendations(
    district_code: int,
    district_name: str,
    date_str: str,
    baseline_row: pd.DataFrame,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze the baseline row to determine the primary stress drivers and evaluate
    potential interventions to rank and recommend the best strategies.
    """
    # 1. Evaluate WSI components to find the primary driver
    def _get_val(col: str) -> float:
        if col in baseline_row.columns:
            val = baseline_row[col].values[0]
            return float(val) if pd.notna(val) else 0.5
        return 0.5

    components = {
        "Groundwater Table Depletion": _get_val("groundwater_stress_score"),
        "Rainfall Deficit": _get_val("rainfall_stress_score"),
        "Atmospheric Temperature Stress": _get_val("temperature_stress_score"),
        "Humidity Deficit": _get_val("humidity_stress_score"),
        "Surface River Deficit": _get_val("river_level_stress_score"),
    }
    
    primary_driver = max(components, key=components.get)
    driver_severity = components[primary_driver]

    # 2. Run mini-simulations for each intervention at a test intensity of 0.5 to rank them
    # Baseline WSI predicted at 30 days
    baseline_predictions = predict_water_stress_scenarios(
        baseline_row, models, 0.0, 0.0, 0.0, 0.0
    )
    baseline_30d = float(baseline_predictions["wsi_lead_30"][0])

    # Sim 1: Rainwater Harvesting only
    rwh_preds = predict_water_stress_scenarios(baseline_row, models, 0.5, 0.0, 0.0, 0.0)
    rwh_reduction = baseline_30d - float(rwh_preds["wsi_lead_30"][0])

    # Sim 2: Demand Reduction only
    dr_preds = predict_water_stress_scenarios(baseline_row, models, 0.0, 0.5, 0.0, 0.0)
    dr_reduction = baseline_30d - float(dr_preds["wsi_lead_30"][0])

    # Sim 3: Water Conservation only
    wc_preds = predict_water_stress_scenarios(baseline_row, models, 0.0, 0.0, 0.5, 0.0)
    wc_reduction = baseline_30d - float(wc_preds["wsi_lead_30"][0])

    # Sim 4: Additional Water Supply only
    aws_preds = predict_water_stress_scenarios(baseline_row, models, 0.0, 0.0, 0.0, 0.5)
    aws_reduction = baseline_30d - float(aws_preds["wsi_lead_30"][0])

    # Rank the interventions
    reductions = [
        ("Rainwater Harvesting", rwh_reduction, "Implement rainwater collection on public buildings and residential societies to recharge the aquifers."),
        ("Demand Reduction", dr_reduction, "Deploy smart meters, enforce dynamic pricing, and restrict water supply during peak hours for non-essential use."),
        ("Water Conservation", wc_reduction, "Conduct awareness campaigns, encourage low-flow bathroom fixtures, and incentivize drip irrigation in surrounding rural areas."),
        ("Additional Water Supply", aws_reduction, "Increase regional raw water allocations, treat municipal water to drinking standards, and optimize pipeline supply distribution.")
    ]
    
    reductions.sort(key=lambda x: x[1], reverse=True)
    top_strategy, top_reduction, top_action = reductions[0]

    # 3. Create descriptive strategies
    actions_mapping = {
        "Groundwater Table Depletion": [
            "Mandate artificial groundwater recharge injection wells in industrial zones.",
            "Promote water auditing for high-consumption commercial buildings.",
            "Enforce strict groundwater extraction quotas and licensing."
        ],
        "Rainwater Deficit": [
            "Subsidize rooftop rainwater harvesting systems in residential wards.",
            "Create check dams and urban wetlands to maximize storm runoff retention.",
            "Upgrade drainage infrastructure to capture stormwater."
        ],
        "Atmospheric Temperature Stress": [
            "Implement rooftop vegetation and urban greening to combat urban heat island effects.",
            "Schedule municipal water supply shifts during cooler evening hours to prevent evaporation losses.",
            "Encourage agricultural shading and micro-sprinklers in surrounding peri-urban zones."
        ],
        "Humidity Deficit": [
            "Increase canopy coverage in dry sectors to modify local micro-climates.",
            "Adopt vertical gardens and green walls in high-density wards."
        ],
        "Surface River Deficit": [
            "Coordinate with regional reservoir managers to schedule controlled canal releases.",
            "Enforce strict surface discharge treatment to allow direct recycling of wastewater into rivers."
        ]
    }

    selected_actions = actions_mapping.get(primary_driver, [
        "Implement localized water supply recycling.",
        "Promote micro-irrigation systems."
    ])

    return {
        "district_code": district_code,
        "district_name": district_name,
        "date": date_str,
        "baseline_wsi_30d": baseline_30d,
        "primary_driver": primary_driver,
        "driver_severity": driver_severity,
        "component_stresses": components,
        "recommended_strategy": top_strategy,
        "expected_30d_reduction": max(0.0, top_reduction),
        "primary_action": top_action,
        "action_items": selected_actions,
        "strategy_rankings": [
            {"strategy": r[0], "expected_reduction": float(max(0.0, r[1]))} for r in reductions
        ]
    }
