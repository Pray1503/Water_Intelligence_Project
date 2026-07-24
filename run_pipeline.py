import subprocess
import sys
import logging
from pathlib import Path

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    project_root = Path(__file__).resolve().parent
    discover_script = project_root / 'scripts' / '01_discover_datasets.py'
    
    try:
        # Run Phase 1A: Dataset Discovery
        subprocess.run([sys.executable, str(discover_script)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error(f"Could not find the script at {discover_script}")
        sys.exit(1)

if __name__ == "__main__":
    main()
