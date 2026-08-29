"""Profiling mode selector and pushdown SQL profiler.

Adaptive profiling strategy:
- Mode A (Full Pushdown): SQL aggregates for all columns. Used when row count <= threshold.
- Mode B (Hybrid): Full aggregates for numeric/date + sampled string profiling for high-cardinality columns.
- Mode C (Progressive Sampling): Multi-pass sampling until metric stability.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.models import ColumnInfo, ColumnProfile, TableMetadata, TableProfile


_NUMERIC_BASES = {
    "int", "bigint", "smallint", "tinyint", "decimal",
    "numeric", "float", "real", "money", "smallmoney",
}

_BOOL_BASES = {"bit"}

# Max columns per stats batch — keeps SELECT list width manageable.
_STATS_BATCH_SIZE = 150

# Row count above which TABLESAMPLE is applied for Mode B/C tables.
# At or below this threshold the full table is scanned (Mode A).
_TABLESAMPLE_THRESHOLD = 500_000

# Number of rows to sample when TABLESAMPLE is active.
# 50 000 rows provides stable cardinality estimates for typical datasets.
_TABLESAMPLE_ROWS = 50_000


def _base_type(raw: str) -> str:
    """Strip type modifiers and lowercase."""
    idx = raw.find("(")
    return raw[:idx].strip().lower() if idx >= 0 else raw.strip().lower()


def _quote_identifier(name: str) -> str:
    """Quote a SQL Server identifier with brackets."""
    safe = name.replace("]", "]]")
    return f"[{safe}]"


def _wants_minmax(col: ColumnInfo) -> bool:
    """Return True for column types that support MIN/MAX aggregation.

    SQL Server `bit` is excluded (error 8117); binary, xml, spatial excluded too.
    """
    base = _base_type(col.data_type)
    return (base in _NUMERIC_BASES and base not in _BOOL_BASES) or base in (
        "date", "time", "datetime", "datetime2", "smalldatetime", "datetimeoffset",
    )


def select_profiling_mode(
    row_count: int,
    string_columns: int,
    mode_a_max_rows: int = 100000,
    mode_b_string_cardinality: int = 5000,
) -> str:
    """Select the profiling mode for a table.

    Args:
        row_count: Total rows in the table.
        string_columns: Number of string-type columns.
        mode_a_max_rows: Max rows for Mode A (full pushdown).
        mode_b_string_cardinality: Threshold for string column sampling in Mode B.

    Returns:
        "A", "B", or "C".
    """
    if row_count <= mode_a_max_rows:
        return "A"
    if string_columns <= 20:
        return "B"
    return "C"


def _run_stats_query(
    conn: Any,
    schema: str,
    table: str,
    columns: List[ColumnInfo],
    tablesample_rows: Optional[int] = None,
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """One consolidated SELECT to get null/distinct/min/max for every column.

    Processes columns in batches of _STATS_BATCH_SIZE to bound SELECT-list width.
    The first batch includes COUNT_BIG(*) to retrieve the row count in the same
    round trip.

    Args:
        conn: Open pyodbc connection.
        schema: Schema name.
        table: Table name.
        columns: Columns to profile.
        tablesample_rows: When set, appends ``TABLESAMPLE (N ROWS)`` to the
            FROM clause so SQL Server works on a bounded sample rather than
            the full table.  Metrics become approximate; use for Mode B/C.

    Returns:
        (row_count, {col_name: {non_null, null_count, distinct, min_val, max_val}})
    """
    quoted_table = f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
    sample_clause = (
        f" TABLESAMPLE ({tablesample_rows} ROWS)" if tablesample_rows else ""
    )
    stats: Dict[str, Dict[str, Any]] = {}
    row_count = 0

    for batch_start in range(0, len(columns), _STATS_BATCH_SIZE):
        batch = columns[batch_start : batch_start + _STATS_BATCH_SIZE]
        first_batch = batch_start == 0

        parts: List[str] = []
        if first_batch:
            parts.append("COUNT_BIG(*)")

        for col in batch:
            qc = _quote_identifier(col.name)
            minmax = _wants_minmax(col)
            parts.extend([
                f"COUNT_BIG({qc})",
                f"COUNT_BIG(CASE WHEN {qc} IS NULL THEN 1 END)",
                f"COUNT(DISTINCT {qc})",
                f"MIN({qc})" if minmax else "NULL",
                f"MAX({qc})" if minmax else "NULL",
            ])

        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {', '.join(parts)} FROM {quoted_table}{sample_clause}"
        )
        row = cursor.fetchone()
        cursor.close()

        offset = 0
        if first_batch:
            row_count = int(row[0] or 0)
            offset = 1

        for i, col in enumerate(batch):
            base = offset + i * 5
            stats[col.name] = {
                "non_null":   int(row[base]     or 0),
                "null_count": int(row[base + 1] or 0),
                "distinct":   int(row[base + 2] or 0),
                "min_val":    row[base + 3],
                "max_val":    row[base + 4],
            }

    return row_count, stats


def _fetch_top_values(
    conn: Any,
    quoted_table: str,
    quoted_col: str,
    top_n: int,
    tablesample_rows: Optional[int] = None,
) -> List[Any]:
    """Fetch the most frequent non-null values for one column.

    Args:
        conn: Open pyodbc connection.
        quoted_table: Bracket-quoted ``[schema].[table]``.
        quoted_col: Bracket-quoted column name.
        top_n: Maximum number of top values to return.
        tablesample_rows: When set, uses ``TABLESAMPLE (N ROWS)`` so the
            GROUP BY runs over a sample instead of the full table.
    """
    sample_clause = (
        f" TABLESAMPLE ({tablesample_rows} ROWS)" if tablesample_rows else ""
    )
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT TOP ({top_n}) {quoted_col}, COUNT_BIG(*) AS cnt "
            f"FROM {quoted_table}{sample_clause} "
            f"WHERE {quoted_col} IS NOT NULL "
            f"GROUP BY {quoted_col} "
            f"ORDER BY cnt DESC"
        )
        return [r[0] for r in cursor.fetchall()]
    except Exception:
        return []
    finally:
        cursor.close()


def _run_column_profiles_pushdown(
    conn: Any,
    schema: str,
    table: str,
    columns: List[ColumnInfo],
    top_n: int = 10,
    tablesample_rows: Optional[int] = None,
) -> Dict[str, ColumnProfile]:
    """Profile all columns: one consolidated stats query + sequential top-value fetches.

    Args:
        conn: Open pyodbc connection.
        schema: Schema name.
        table: Table name.
        columns: List of columns to profile.
        top_n: Number of top values to capture per column.
        tablesample_rows: When set, all queries use ``TABLESAMPLE (N ROWS)``
            so metrics are approximate but fast for large tables.

    Returns:
        Dictionary mapping column name -> ColumnProfile.
    """
    quoted_table = f"{_quote_identifier(schema)}.{_quote_identifier(table)}"

    row_count, stats = _run_stats_query(
        conn, schema, table, columns, tablesample_rows=tablesample_rows
    )

    top_values_map: Dict[str, List[Any]] = {
        col.name: _fetch_top_values(
            conn,
            quoted_table,
            _quote_identifier(col.name),
            top_n,
            tablesample_rows=tablesample_rows,
        )
        for col in columns
    }

    profiles: Dict[str, ColumnProfile] = {}
    for col in columns:
        s = stats.get(col.name, {})
        non_null   = s.get("non_null",   0)
        null_count = s.get("null_count", 0)
        distinct   = s.get("distinct",   0)

        null_ratio       = null_count / row_count if row_count > 0 else 0.0
        distinct_ratio   = distinct   / non_null  if non_null  > 0 else 0.0
        uniqueness_ratio = distinct   / row_count if row_count > 0 else 0.0

        profiles[col.name] = ColumnProfile(
            name=col.name,
            data_type=col.data_type,
            row_count=row_count,
            non_null_count=non_null,
            null_count=null_count,
            null_ratio=round(null_ratio, 6),
            distinct_count=distinct,
            distinct_ratio=round(distinct_ratio, 6),
            uniqueness_ratio=round(uniqueness_ratio, 6),
            min_value=s.get("min_val"),
            max_value=s.get("max_val"),
            top_values=top_values_map.get(col.name, []),
            profiling_mode="A",
        )

    return profiles


def profile_table(
    conn: Any,
    metadata: TableMetadata,
    config: Optional[Dict[str, Any]] = None,
) -> TableProfile:
    """Profile a single table using the appropriate profiling mode.

    Args:
        conn: Open pyodbc connection.
        metadata: Full table metadata from inventory.
        config: Configuration dictionary with profiling thresholds.

    Returns:
        TableProfile with per-column profiles.
    """
    if config is None:
        config = {}

    profiling_cfg = config.get("profiling", {})
    mode_a_max         = profiling_cfg.get("mode_a_max_rows",         100000)
    mode_b_string_card = profiling_cfg.get("mode_b_string_cardinality", 5000)
    ts_threshold       = profiling_cfg.get("tablesample_threshold",  _TABLESAMPLE_THRESHOLD)
    ts_rows            = profiling_cfg.get("tablesample_rows",        _TABLESAMPLE_ROWS)

    string_cols = [
        c for c in metadata.columns
        if _base_type(c.data_type) in ("varchar", "nvarchar", "char", "nchar", "text", "ntext")
    ]
    mode = select_profiling_mode(
        metadata.row_count, len(string_cols), mode_a_max, mode_b_string_card
    )

    # For large tables (Mode B/C) use TABLESAMPLE so COUNT(DISTINCT) and
    # GROUP BY top-value queries work over a bounded sample rather than the
    # full table.  Metrics become approximate but load time drops dramatically.
    # Set tablesample_threshold: 0 in config to disable.
    tablesample_rows: Optional[int] = None
    if ts_threshold > 0 and metadata.row_count > ts_threshold and mode in ("B", "C"):
        tablesample_rows = ts_rows

    columns = _run_column_profiles_pushdown(
        conn,
        metadata.schema_name,
        metadata.table_name,
        metadata.columns,
        tablesample_rows=tablesample_rows,
    )

    _mode_notes = {
        "A": "Full pushdown profiling",
        "B": "Hybrid profiling (full aggregates + sampled strings)",
        "C": "Progressive sampling mode (using pushdown baseline)",
    }

    return TableProfile(
        schema_name=metadata.schema_name,
        table_name=metadata.table_name,
        row_count=metadata.row_count,
        column_count=len(metadata.columns),
        columns=columns,
        profiling_mode=mode,
        profiling_note=_mode_notes.get(mode, ""),
    )


def profile_tables(
    conn: Any,
    inventory: Dict[str, TableMetadata],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, TableProfile]:
    """Profile multiple tables sequentially.

    Args:
        conn: Open pyodbc connection.
        inventory: Dictionary of table metadata keyed by `{schema}.{table}`.
        config: Optional configuration.

    Returns:
        Dictionary mapping `{schema}.{table}` to TableProfile.
    """
    return {
        key: profile_table(conn, metadata, config)
        for key, metadata in inventory.items()
    }
