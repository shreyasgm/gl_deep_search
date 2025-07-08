"""
Entry point for running the ETL orchestrator as a module.

This allows running the orchestrator with:
    python -m backend.etl
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.etl.orchestrator import main  # noqa: E402

if __name__ == "__main__":
    asyncio.run(main())
