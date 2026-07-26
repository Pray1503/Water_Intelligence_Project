import logging
import sys
from pathlib import Path
import json

# Remove the script directory from sys.path to prevent shadowing standard libraries (like inspect.py)
script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != script_dir]

import pandas as pd
import yaml
from typing import Dict, Any

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def profile_single_dataset(abs_path: Path) -> Dict[str, Any]:
    """Profile a single dataset using chunked reading."""
    total_rows = 0
    columns = []
    dtypes = {}
    
    missing_counts = {}
    unique_sets = {}
    numeric_stats = {}
    
    # Read in chunks to handle large files
    try:
        for i, chunk in enumerate(pd.read_csv(abs_path, chunksize=100000, low_memory=False)):
            if i == 0:
                columns = chunk.columns.tolist()
                dtypes = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
                for col in columns:
                    missing_counts[col] = 0
                    unique_sets[col] = set()
                    
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        numeric_stats[col] = {
                            "min": float("inf"),
                            "max": float("-inf"),
                            "sum": 0.0,
                            "sum_sq": 0.0,
                            "non_null_count": 0
                        }
            
            total_rows += len(chunk)
            
            for col in columns:
                if col in chunk.columns:
                    col_data = chunk[col]
                    missing_counts[col] += int(col_data.isna().sum())
                    unique_sets[col].update(col_data.dropna().unique())
                    
                    if col in numeric_stats:
                        valid_data = col_data.dropna()
                        if len(valid_data) > 0:
                            stats = numeric_stats[col]
                            stats["min"] = min(stats["min"], float(valid_data.min()))
                            stats["max"] = max(stats["max"], float(valid_data.max()))
                            stats["sum"] += float(valid_data.sum())
                            stats["sum_sq"] += float((valid_data ** 2).sum())
                            stats["non_null_count"] += len(valid_data)
                            
    except Exception as e:
        logger.error(f"Error profiling {abs_path.name}: {e}")
        return {}

    column_profiles = []
    for col in columns:
        missing_count = missing_counts.get(col, 0)
        missing_percentage = (missing_count / total_rows * 100) if total_rows > 0 else 0.0
        unique_count = len(unique_sets.get(col, set()))
        
        col_profile = {
            "Name": col,
            "Data Type": dtypes.get(col, "Unknown"),
            "Missing Value Count": missing_count,
            "Missing Value Percentage": round(missing_percentage, 2),
            "Unique Value Count": unique_count
        }
        
        if col in numeric_stats:
            stats = numeric_stats[col]
            n = stats["non_null_count"]
            if n > 0:
                col_profile["Minimum"] = stats["min"]
                col_profile["Maximum"] = stats["max"]
                
                mean = stats["sum"] / n
                col_profile["Mean"] = round(mean, 4)
                
                if n > 1:
                    variance = (stats["sum_sq"] - (stats["sum"] ** 2) / n) / (n - 1)
                    variance = max(0.0, variance)
                    col_profile["Standard Deviation"] = round(variance ** 0.5, 4)
                else:
                    col_profile["Standard Deviation"] = 0.0
                
                col_profile["Non-Null Count"] = n
                
        column_profiles.append(col_profile)

    return {
        "Total Rows": total_rows,
        "Total Columns": len(columns),
        "Column Profiles": column_profiles
    }

def profile_datasets(config: Dict[str, Any], project_root: Path) -> None:
    """Read the dataset inventory and profile each dataset (Phase 1B Sprint 2)."""
    paths = config.get('paths', {})
    inventory_path_str = paths.get('dataset_inventory', 'reports/dataset_inventory.csv')
    profiling_dir_str = paths.get('profiling_dir', 'reports/profiling')
    
    inventory_path = project_root / inventory_path_str
    profiling_dir = project_root / profiling_dir_str
    
    if not inventory_path.exists():
        logger.error(f"Inventory file missing at {inventory_path}. Run Phase 1A first.")
        sys.exit(1)

    try:
        df = pd.read_csv(inventory_path)
    except pd.errors.EmptyDataError:
        logger.error("Dataset inventory is empty. No datasets to profile.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read inventory file: {e}")
        sys.exit(1)

    if df.empty:
        logger.error("Dataset inventory is empty. No datasets to profile.")
        sys.exit(1)

    logger.info("Starting dataset profiling...")
    profiling_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        name = row.get("Dataset Name", "Unknown")
        category = row.get("Dataset Category", "Unknown")
        rel_path = row.get("Relative Path", "Unknown")
        abs_path_str = row.get("Absolute Path", "")
        size = row.get("File Size", "Unknown")
        
        logger.info(f"Dataset Name: {name}")
        logger.info(f"Dataset Category: {category}")
        logger.info(f"Relative Path: {rel_path}")
        logger.info(f"File Size: {size}")
        
        if not abs_path_str:
            logger.warning(f"Skipping {name}: Absolute Path missing in inventory.")
            logger.info("-" * 40)
            continue
            
        abs_path = Path(abs_path_str)
        if not abs_path.exists():
            logger.warning(f"Skipping {name}: File not found at {abs_path}")
            logger.info("-" * 40)
            continue
            
        logger.info(f"Profiling {name}...")
        profile_data = profile_single_dataset(abs_path)
        
        if profile_data:
            full_profile = {
                "Profiler": {
                    "Version": "1.0",
                    "Processing Mode": "Chunked"
                },
                "Metadata": {
                    "Dataset Name": name,
                    "Dataset Category": category,
                    "Relative Path": rel_path,
                    "File Size": size
                },
                "Profile": profile_data
            }
            
            out_json = profiling_dir / f"{name}.json"
            try:
                with open(out_json, 'w') as f:
                    json.dump(full_profile, f, indent=4)
                logger.info(f"Saved profile to {out_json.relative_to(project_root).as_posix()}")
            except Exception as e:
                logger.error(f"Failed to save profile for {name}: {e}")
                
        logger.info("-" * 40)

    logger.info("Profiling completed successfully.")

def main():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config' / 'config.yaml'
    
    config = load_config(config_path)
    profile_datasets(config, project_root)

if __name__ == "__main__":
    main()
