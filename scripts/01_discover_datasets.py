import logging
import sys
from pathlib import Path

# Remove the script directory from sys.path to prevent shadowing standard libraries (like inspect.py)
script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != script_dir]

from datetime import datetime
import pandas as pd
import yaml
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format to match requested output
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

def get_dataset_category(filename: str) -> str:
    """Infer the dataset category from the filename."""
    filename_lower = filename.lower()
    if 'gwl' in filename_lower:
        return 'Groundwater'
    elif 'humid' in filename_lower:
        return 'Humidity'
    elif 'rainfall' in filename_lower:
        return 'Rainfall'
    elif 'rwl' in filename_lower:
        return 'River Level'
    elif 'temperature' in filename_lower:
        return 'Temperature'
    return 'Unknown'

def get_file_metadata(file_path: Path, raw_dir: Path) -> Dict[str, Any]:
    """Extract metadata from a single file."""
    stat = file_path.stat()
    return {
        "Dataset Name": file_path.stem,
        "Dataset Category": get_dataset_category(file_path.name),
        "Relative Path": file_path.relative_to(raw_dir.parent).as_posix(),
        "Absolute Path": file_path.resolve().as_posix(),
        "File Size": stat.st_size,
        "Extension": file_path.suffix,
        "Last Modified Timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "Parent Folder": file_path.parent.name,
        "Status": "RAW"
    }

def discover_datasets(config: Dict[str, Any], project_root: Path) -> None:
    """Discover all CSV datasets in the raw data directory."""
    paths = config.get('paths', {})
    
    # Read paths from config
    raw_dir_str = paths.get('data_raw', 'data/raw')
    reports_dir_str = paths.get('reports_dir', 'reports')
    inventory_path_str = paths.get('dataset_inventory', 'reports/dataset_inventory.csv')
    
    raw_dir = project_root / raw_dir_str
    reports_dir = project_root / reports_dir_str
    inventory_path = project_root / inventory_path_str

    logger.info(f"Scanning {raw_dir_str}...")

    if not raw_dir.exists():
        logger.warning(f"Raw data directory {raw_dir} does not exist.")
        return

    metadata_list: List[Dict[str, Any]] = []

    # Recursively scan for CSV files
    for file_path in raw_dir.rglob("*.csv"):
        logger.info(f"Found {file_path.name}")
        try:
            metadata = get_file_metadata(file_path, raw_dir)
            metadata_list.append(metadata)
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")

    if not metadata_list:
        logger.info("No CSV datasets found.")
    else:
        logger.info("Saving dataset inventory...")
        # Ensure reports directory exists
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save inventory to CSV
        df = pd.DataFrame(metadata_list)
        df.to_csv(inventory_path, index=False)
    
    logger.info("Discovery completed successfully.")

def main():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config' / 'config.yaml'
    
    config = load_config(config_path)
    discover_datasets(config, project_root)

if __name__ == "__main__":
    main()
