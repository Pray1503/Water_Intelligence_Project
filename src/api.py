"""
Stage 11: FastAPI Backend API.

Exposes REST endpoints for prediction, scenario simulation, and prescriptive recommendations.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_float(val, default: float = 0.5) -> float:
    """Safely convert a value to a float, returning a default if NaN, infinite, or invalid."""
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return f_val
    except Exception:
        return default

app = FastAPI(
    title="Water Intelligence Platform API",
    description="AI-powered decision support API for predicting water stress and simulating policy interventions.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to cache dataset and models
DATASET: Optional[pd.DataFrame] = None
MODELS: Dict[str, Any] = {}


@app.on_event("startup")
def startup_event():
    """Load the feature dataset and verify model paths on API startup."""
    global DATASET
    dataset_path = PROJECT_ROOT / "data" / "features" / "feature_dataset_with_labels.parquet"
    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Please run Stage 8 label generation.", dataset_path)
        raise RuntimeError(f"Dataset missing: {dataset_path}")
        
    logger.info("Caching feature dataset from: %s", dataset_path)
    DATASET = pd.read_parquet(dataset_path)
    
    # Parse dates to string/pandas datetime for easy comparison
    DATASET["Date"] = pd.to_datetime(DATASET["Date"])
    logger.info("Dataset cached successfully. Shape: %s", DATASET.shape)


def get_models() -> Dict[str, Any]:
    """Lazily load models into memory."""
    global MODELS
    if not MODELS:
        models_dir = PROJECT_ROOT / "models"
        model_paths = {
            "wsi_lead_7": models_dir / "model_7d.joblib",
            "wsi_lead_15": models_dir / "model_15d.joblib",
            "wsi_lead_30": models_dir / "model_30d.joblib"
        }
        
        for key, path in model_paths.items():
            if not path.exists():
                logger.error("Model binary not found at %s. Please run Stage 9 training first.", path)
                raise FileNotFoundError(f"Model missing: {path}")
            logger.info("Loading model binary: %s", path.name)
            MODELS[key] = joblib.load(path)
            
    return MODELS


# Pydantic classes for API validation
class SimulationRequest(BaseModel):
    district_code: int = Field(..., example=438, description="LGD Code of the district.")
    start_date: str = Field(..., example="2025-01-01", description="Start date (YYYY-MM-DD).")
    end_date: str = Field(..., example="2025-03-01", description="End date (YYYY-MM-DD).")
    rainwater_harvesting: float = Field(0.0, ge=0.0, le=1.0, description="Intensity of Rainwater Harvesting (0-1).")
    demand_reduction: float = Field(0.0, ge=0.0, le=1.0, description="Intensity of Demand Reduction (0-1).")
    water_conservation: float = Field(0.0, ge=0.0, le=1.0, description="Intensity of Water Conservation (0-1).")
    additional_water_supply: float = Field(0.0, ge=0.0, le=1.0, description="Intensity of Additional Water Supply (0-1).")


# Endpoints
@app.get("/api/districts")
def get_districts():
    """Return a list of available districts and their LGD codes."""
    if DATASET is None:
        raise HTTPException(status_code=503, detail="Dataset not cached.")
        
    districts = (
        DATASET[["District LGD Code", "District"]]
        .drop_duplicates()
        .sort_values(by="District")
        .to_dict(orient="records")
    )
    return {"status": "SUCCESS", "districts": districts}


@app.get("/api/predict")
def get_predictions(
    district_code: int = Query(..., description="LGD Code of the district"),
    start_date: str = Query("2025-01-01", description="Start date (YYYY-MM-DD)"),
    end_date: str = Query("2025-03-31", description="End date (YYYY-MM-DD)")
):
    """Retrieve historical WSI and baseline 7, 15, and 30-day WSI predictions."""
    if DATASET is None:
        raise HTTPException(status_code=503, detail="Dataset not cached.")

    # Filter data
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    mask = (
        (DATASET["District LGD Code"] == district_code) & 
        (DATASET["Date"] >= start_dt) & 
        (DATASET["Date"] <= end_dt)
    )
    subset = DATASET[mask].sort_values("Date")
    if subset.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected parameters.")

    # Load models
    try:
        models = get_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prediction models: {e}")

    # Generate predictions on the fly
    from src.simulation import predict_water_stress_scenarios
    
    records = []
    for _, row in subset.iterrows():
        row_df = pd.DataFrame([row])
        preds = predict_water_stress_scenarios(row_df, models, 0.0, 0.0, 0.0)
        
        records.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "district": row["District"],
            "district_code": int(row["District LGD Code"]),
            "wsi_actual": float(row["wsi"]),
            "pred_wsi_7d": float(preds["wsi_lead_7"][0]),
            "pred_wsi_15d": float(preds["wsi_lead_15"][0]),
            "pred_wsi_30d": float(preds["wsi_lead_30"][0]),
            "components": {
                "groundwater": float(row.get("groundwater_stress_score", 0.5)),
                "rainfall": float(row.get("rainfall_stress_score", 0.5)),
                "temperature": float(row.get("temperature_stress_score", 0.5)),
                "humidity": float(row.get("humidity_stress_score", 0.5)),
                "river_level": float(row.get("river_level_stress_score", 0.5)),
            }
        })

    return {"status": "SUCCESS", "predictions": records}


@app.post("/api/simulate")
def post_simulation(req: SimulationRequest):
    """Simulate intervention policies and return compared predictions."""
    if DATASET is None:
        raise HTTPException(status_code=503, detail="Dataset not cached.")

    try:
        start_dt = pd.to_datetime(req.start_date)
        end_dt = pd.to_datetime(req.end_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    mask = (
        (DATASET["District LGD Code"] == req.district_code) & 
        (DATASET["Date"] >= start_dt) & 
        (DATASET["Date"] <= end_dt)
    )
    subset = DATASET[mask].sort_values("Date")
    if subset.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected parameters.")

    try:
        models = get_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prediction models: {e}")

    from src.simulation import predict_water_stress_scenarios

    records = []
    for _, row in subset.iterrows():
        row_df = pd.DataFrame([row])
        
        # 1. Baseline predictions
        baseline = predict_water_stress_scenarios(row_df, models, 0.0, 0.0, 0.0, 0.0)
        
        # 2. Simulated predictions
        simulated = predict_water_stress_scenarios(
            row_df, models, req.rainwater_harvesting, req.demand_reduction, req.water_conservation, req.additional_water_supply
        )
        
        records.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "baseline": {
                "wsi_7d": float(baseline["wsi_lead_7"][0]),
                "wsi_15d": float(baseline["wsi_lead_15"][0]),
                "wsi_30d": float(baseline["wsi_lead_30"][0])
            },
            "simulated": {
                "wsi_7d": float(simulated["wsi_lead_7"][0]),
                "wsi_15d": float(simulated["wsi_lead_15"][0]),
                "wsi_30d": float(simulated["wsi_lead_30"][0])
            }
        })

    return {
        "status": "SUCCESS",
        "district_code": req.district_code,
        "interventions": {
            "rainwater_harvesting": req.rainwater_harvesting,
            "demand_reduction": req.demand_reduction,
            "water_conservation": req.water_conservation,
            "additional_water_supply": req.additional_water_supply
        },
        "simulation": records
    }


@app.get("/api/recommend")
def get_recommend(
    district_code: int = Query(..., description="LGD Code of the district"),
    date: str = Query(..., description="Target date (YYYY-MM-DD)")
):
    """Return strategy recommendations and predicted benefits for a specific district and date."""
    if DATASET is None:
        raise HTTPException(status_code=503, detail="Dataset not cached.")

    try:
        target_dt = pd.to_datetime(date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    mask = (DATASET["District LGD Code"] == district_code) & (DATASET["Date"] == target_dt)
    subset = DATASET[mask]
    if subset.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected district and date.")

    try:
        models = get_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load prediction models: {e}")

    from src.simulation import generate_recommendations
    
    rec_result = generate_recommendations(
        district_code=district_code,
        district_name=subset["District"].values[0],
        date_str=date,
        baseline_row=subset,
        models=models
    )
    
    return {"status": "SUCCESS", "recommendation": rec_result}
