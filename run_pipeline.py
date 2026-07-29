import subprocess
import sys
import logging
from pathlib import Path

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    project_root = Path(__file__).resolve().parent
    discover_script = project_root / "scripts" / "01_discover_datasets.py"
    profile_script = project_root / "scripts" / "02_profile_datasets.py"
    validate_script = project_root / "scripts" / "03_validate_schemas.py"
    clean_script = project_root / "scripts" / "04_clean_datasets.py"
    audit_script = project_root / "scripts" / "05_dataset_audit.py"
    build_master_dataset_script = (
        project_root / "scripts" / "06_build_master_dataset.py"
    )

    try:
        # Run Phase 1A: Dataset Discovery
        subprocess.run([sys.executable, str(discover_script)], check=True)

        # Run Phase 1B Sprint 1: Dataset Profiling Setup
        subprocess.run([sys.executable, str(profile_script)], check=True)

        # Run Phase 1C Sprint 1: Schema Validation
        subprocess.run([sys.executable, str(validate_script)], check=True)

        # Run Stage 4: Data Cleaning & Consolidation
        subprocess.run([sys.executable, str(clean_script)], check=True)

        # Run Stage 5: Dataset Audit & Quality Reporting
        subprocess.run([sys.executable, str(audit_script)], check=True)

        # Run Stage 6: Master Dataset Builder
        subprocess.run([sys.executable, str(build_master_dataset_script)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logging.error(f"Could not find the script: {e.filename}")
        sys.exit(1)


if __name__ == "__main__":
    main()
