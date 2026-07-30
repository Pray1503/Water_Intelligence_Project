"""
Start FastAPI Backend server.
"""

from __future__ import annotations

import sys
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    print("=" * 70)
    print("STARTING FASTAPI BACKEND SERVER")
    print("Access the API Swagger documentation at: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
