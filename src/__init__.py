"""SQL Relationship Builder.

Deterministic relationship analysis engine with interactive Streamlit curation UI.
"""

from src.models import (
    ColumnInfo,
    ColumnProfile,
    ForeignKeyInfo,
    IndexInfo,
    StringProfile,
    TableInfo,
    TableMetadata,
    TableProfile,
)
from src.types import canonicalize_type, CanonicalType
from src.graph import RelationshipGraph, SuggestedEdge, ConfirmedEdge
from src.pipeline import run_analysis, AnalysisResult

__all__ = [
    "ColumnInfo",
    "ColumnProfile",
    "ForeignKeyInfo",
    "IndexInfo",
    "StringProfile",
    "TableInfo",
    "TableMetadata",
    "TableProfile",
    "canonicalize_type",
    "CanonicalType",
    "RelationshipGraph",
    "SuggestedEdge",
    "ConfirmedEdge",
    "run_analysis",
    "AnalysisResult",
]