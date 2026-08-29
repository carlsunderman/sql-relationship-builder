"""Type canonicalization using sqlglot.

Maps native SQL Server types to canonical families for cross-table
type compatibility evaluation.
"""

from enum import Enum
from typing import Dict, Optional

import sqlglot


class CanonicalType(Enum):
    """Canonical type families for type compatibility."""
    NUMERIC = "numeric"
    STRING = "string"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    BINARY = "binary"
    JSON = "json"
    UNKNOWN = "unknown"
    SPATIAL = "spatial"
    OTHER = "other"


# Fallback map: SQL Server type patterns -> canonical type
# Used when sqlglot cannot determine the type or is not available.
FALLBACK_TYPE_MAP: Dict[str, CanonicalType] = {
    # Exact numeric
    "int": CanonicalType.NUMERIC,
    "bigint": CanonicalType.NUMERIC,
    "smallint": CanonicalType.NUMERIC,
    "tinyint": CanonicalType.NUMERIC,
    "bit": CanonicalType.BOOLEAN,
    "decimal": CanonicalType.NUMERIC,
    "numeric": CanonicalType.NUMERIC,
    "money": CanonicalType.NUMERIC,
    "smallmoney": CanonicalType.NUMERIC,
    "float": CanonicalType.NUMERIC,
    "real": CanonicalType.NUMERIC,
    # Strings
    "char": CanonicalType.STRING,
    "varchar": CanonicalType.STRING,
    "nchar": CanonicalType.STRING,
    "nvarchar": CanonicalType.STRING,
    "text": CanonicalType.STRING,
    "ntext": CanonicalType.STRING,
    # Date/Time
    "date": CanonicalType.DATETIME,
    "time": CanonicalType.DATETIME,
    "datetime": CanonicalType.DATETIME,
    "datetime2": CanonicalType.DATETIME,
    "smalldatetime": CanonicalType.DATETIME,
    "datetimeoffset": CanonicalType.DATETIME,
    # Binary
    "binary": CanonicalType.BINARY,
    "varbinary": CanonicalType.BINARY,
    "image": CanonicalType.BINARY,
    # JSON
    "json": CanonicalType.JSON,
    # Spatial
    "geometry": CanonicalType.SPATIAL,
    "geography": CanonicalType.SPATIAL,
    # Other
    "uniqueidentifier": CanonicalType.OTHER,
    "xml": CanonicalType.OTHER,
    "sql_variant": CanonicalType.OTHER,
    "hierarchyid": CanonicalType.OTHER,
    "rowversion": CanonicalType.OTHER,
    "timestamp": CanonicalType.OTHER,
}


def _strip_type_modifiers(raw_type: str) -> str:
    """Remove length/precision/scale modifiers from a type string.

    Example: 'nvarchar(255)' -> 'nvarchar', 'decimal(18,2)' -> 'decimal'

    Args:
        raw_type: The raw type string from SQL Server.

    Returns:
        Base type name without modifiers.
    """
    idx = raw_type.find("(")
    return raw_type[:idx].strip() if idx >= 0 else raw_type.strip()


def canonicalize_type(native_type: str) -> CanonicalType:
    """Map a native SQL Server type to its canonical family.

    Uses sqlglot for parsing first, falls back to static map.

    Args:
        native_type: The native type string (e.g., 'nvarchar(255)', 'decimal(18,2)').

    Returns:
        CanonicalType enum value.
    """
    base = _strip_type_modifiers(native_type).lower()

    # Try sqlglot first
    try:
        parsed = sqlglot.expressions.DataType.build(base, dialect="mssql")
        if parsed is not None:
            sqlglot_type = parsed.this
            if sqlglot_type == sqlglot.expressions.DataType.Type.INT:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.BIGINT:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.SMALLINT:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.TINYINT:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.BOOLEAN:
                return CanonicalType.BOOLEAN
            if sqlglot_type == sqlglot.expressions.DataType.Type.DECIMAL:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.DOUBLE:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.FLOAT:
                return CanonicalType.NUMERIC
            if sqlglot_type == sqlglot.expressions.DataType.Type.VARCHAR:
                return CanonicalType.STRING
            if sqlglot_type == sqlglot.expressions.DataType.Type.NVARCHAR:
                return CanonicalType.STRING
            if sqlglot_type == sqlglot.expressions.DataType.Type.CHAR:
                return CanonicalType.STRING
            if sqlglot_type == sqlglot.expressions.DataType.Type.NCHAR:
                return CanonicalType.STRING
            if sqlglot_type == sqlglot.expressions.DataType.Type.TEXT:
                return CanonicalType.STRING
            if sqlglot_type == sqlglot.expressions.DataType.Type.DATE:
                return CanonicalType.DATETIME
            if sqlglot_type == sqlglot.expressions.DataType.Type.DATETIME:
                return CanonicalType.DATETIME
            if sqlglot_type == sqlglot.expressions.DataType.Type.TIMESTAMP:
                return CanonicalType.DATETIME
            if sqlglot_type == sqlglot.expressions.DataType.Type.TIME:
                return CanonicalType.DATETIME
            if sqlglot_type == sqlglot.expressions.DataType.Type.BINARY:
                return CanonicalType.BINARY
            if sqlglot_type == sqlglot.expressions.DataType.Type.VARBINARY:
                return CanonicalType.BINARY
            if sqlglot_type == sqlglot.expressions.DataType.Type.JSON:
                return CanonicalType.JSON
            if sqlglot_type == sqlglot.expressions.DataType.Type.GEOMETRY:
                return CanonicalType.SPATIAL
            if sqlglot_type == sqlglot.expressions.DataType.Type.GEOGRAPHY:
                return CanonicalType.SPATIAL
            if sqlglot_type == sqlglot.expressions.DataType.Type.UNIQUEIDENTIFIER:
                return CanonicalType.OTHER
            if sqlglot_type == sqlglot.expressions.DataType.Type.XML:
                return CanonicalType.OTHER
    except Exception:
        pass

    # Fallback to static map
    return FALLBACK_TYPE_MAP.get(base, CanonicalType.UNKNOWN)


def are_types_compatible(
    type_a: str,
    type_b: str,
    strict: bool = True,
) -> bool:
    """Check if two native SQL Server types are compatible for joining.

    Args:
        type_a: Native type string from table A.
        type_b: Native type string from table B.
        strict: If True, only same canonical family is compatible.
                If False, also allows NUMERIC<->STRING (cast-required).

    Returns:
        True if types are compatible.
    """
    canon_a = canonicalize_type(type_a)
    canon_b = canonicalize_type(type_b)

    if canon_a == canon_b:
        return True

    if not strict:
        # Relaxed: allow numeric-string conversion
        if {canon_a, canon_b} == {CanonicalType.NUMERIC, CanonicalType.STRING}:
            return True

    return False


def type_compatibility_decision(
    type_a: str,
    type_b: str,
    strict: bool = True,
) -> Dict:
    """Evaluate type compatibility with a detailed decision.

    Returns a dict with 'compatible' (bool), 'canonical_a', 'canonical_b',
    and 'risk' (None, 'cast_required', or 'incompatible').

    Args:
        type_a: Native type string from table A.
        type_b: Native type string from table B.
        strict: If True, strict compatibility enforcement.

    Returns:
        Decision dictionary.
    """
    canon_a = canonicalize_type(type_a)
    canon_b = canonicalize_type(type_b)

    if canon_a == canon_b:
        return {
            "compatible": True,
            "canonical_a": canon_a.value,
            "canonical_b": canon_b.value,
            "risk": None,
        }

    if not strict and {canon_a, canon_b} == {CanonicalType.NUMERIC, CanonicalType.STRING}:
        return {
            "compatible": True,
            "canonical_a": canon_a.value,
            "canonical_b": canon_b.value,
            "risk": "cast_required",
        }

    return {
        "compatible": False,
        "canonical_a": canon_a.value,
        "canonical_b": canon_b.value,
        "risk": "incompatible",
    }