"""
Models module for Growth Lab Deep Search ETL components.
"""

# Import models to make them available from backend.etl.models
from backend.etl.models.publications import (
    GrowthLabPublication,
    OpenAlexPublication,
)

__all__ = ["GrowthLabPublication", "OpenAlexPublication"]
