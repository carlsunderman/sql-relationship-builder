"""Data models used across the relationship builder.

All dataclasses needed by multiple modules, defined in one place
with no external dependencies (no pyodbc, no numpy, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnInfo:
    """Schema information for a single column."""
    name: str
    data_type: str
    is_nullable: bool
    ordinal_position: int


@dataclass
class TableInfo:
    """Schema information for a single table."""
    schema_name: str
    table_name: str
    columns: List[ColumnInfo] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"


@dataclass
class ForeignKeyInfo:
    """A foreign key relationship between two tables."""
    source_schema: str
    source_table: str
    source_column: str
    target_schema: str
    target_table: str
    target_column: str
    constraint_name: str = ""


@dataclass
class IndexInfo:
    """Index metadata for a table."""
    index_name: str
    is_primary_key: bool
    is_unique: bool
    column_names: List[str]


@dataclass
class TableMetadata:
    """Full metadata for a table, combining schema + inventory."""
    schema_name: str
    table_name: str
    row_count: int
    columns: List[ColumnInfo]
    indexes: List[IndexInfo] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def primary_key_columns(self) -> List[str]:
        pks: List[str] = []
        for idx in self.indexes:
            if idx.is_primary_key:
                pks.extend(idx.column_names)
        return pks


@dataclass
class ColumnProfile:
    """Profiling results for a single column."""
    name: str
    data_type: str
    row_count: int
    non_null_count: int
    null_count: int
    null_ratio: float
    distinct_count: int
    distinct_ratio: float
    uniqueness_ratio: float
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    top_values: List[Any] = field(default_factory=list)
    profiling_mode: Optional[str] = "A"
    profiling_note: str = ""


@dataclass
class TableProfile:
    """Profiling results for a table."""
    schema_name: str
    table_name: str
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    profiling_mode: str
    profiling_note: str = ""


@dataclass
class StringProfile:
    """Profiling results specific to string columns."""
    column_name: str
    data_type: str
    total_count: int
    non_null_count: int
    distinct_count: int
    null_ratio: float
    is_categorical: bool
    categorical_distinct_count: int
    is_identifier_like: bool
    avg_length: float
    min_length: int
    max_length: int
    contains_numbers: bool
    contains_special_chars: bool
    whitespace_normalized_distinct: Optional[int] = None
    lower_normalized_distinct: Optional[int] = None
    normalization_reduces: bool = False
    sample_values: List[str] = field(default_factory=list)
    profiling_note: str = ""