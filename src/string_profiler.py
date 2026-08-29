"""String profiler.

Analyzes string-type columns for categorical detection, identifier heuristics,
length distributions, and normalization effects.
"""

import re
from typing import Any, Dict, List, Optional

from src.models import ColumnInfo, StringProfile, TableMetadata, TableProfile


_STRING_STATS_BATCH_SIZE = 100

# Mirror profiler thresholds: sample string stats for large tables.
_TABLESAMPLE_THRESHOLD = 500_000
_TABLESAMPLE_ROWS = 50_000


def _quote_identifier(name: str) -> str:
    """Quote a SQL Server identifier with brackets."""
    safe = name.replace("]", "]]")
    return f"[{safe}]"


# Patterns for identifier detection
_IDENTIFIER_PATTERNS = [
    re.compile(r"^[A-Z0-9]{8,}$"),
    re.compile(r"^[A-Z]{2,}-\d{3,}"),
    re.compile(r"^\d{10,14}$"),
    re.compile(r"^[A-Z]{2}\d{5,}$"),
]

# Column names that suggest categorical data
_CATEGORICAL_HINT_COLUMNS = {
    "status", "type", "category", "class", "region", "state",
    "country", "county", "area", "zone", "code", "phase",
    "unit", "stage", "color", "grade", "lithology",
}


def _run_string_stats_batch(
    conn: Any,
    schema: str,
    table: str,
    string_cols: List[ColumnInfo],
    tablesample_rows: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """One query per batch for length + normalization stats across all string columns.

    Args:
        conn: Open pyodbc connection.
        schema: Schema name.
        table: Table name.
        string_cols: String-type columns to profile.
        tablesample_rows: When set, appends ``TABLESAMPLE (N ROWS)`` so
            AVG/MIN/MAX/DISTINCT aggregates run over a sample for large tables.

    Returns:
        {col_name: {avg_length, min_length, max_length, whitespace_norm_distinct, lower_norm_distinct}}
    """
    quoted_table = f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
    sample_clause = (
        f" TABLESAMPLE ({tablesample_rows} ROWS)" if tablesample_rows else ""
    )
    stats: Dict[str, Dict[str, Any]] = {}

    for batch_start in range(0, len(string_cols), _STRING_STATS_BATCH_SIZE):
        batch = string_cols[batch_start : batch_start + _STRING_STATS_BATCH_SIZE]

        parts: List[str] = []
        for col in batch:
            qc = _quote_identifier(col.name)
            parts.extend([
                f"AVG(LEN({qc}) * 1.0)",
                f"MIN(LEN({qc}))",
                f"MAX(LEN({qc}))",
                f"COUNT(DISTINCT LTRIM(RTRIM({qc})))",
                f"COUNT(DISTINCT LOWER({qc}))",
            ])

        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {', '.join(parts)} FROM {quoted_table}{sample_clause}"
        )
        row = cursor.fetchone()
        cursor.close()

        for i, col in enumerate(batch):
            base = i * 5
            stats[col.name] = {
                "avg_length":               float(row[base])     if row[base]     is not None else 0.0,
                "min_length":               int(row[base + 1])   if row[base + 1] is not None else 0,
                "max_length":               int(row[base + 2])   if row[base + 2] is not None else 0,
                "whitespace_norm_distinct":  int(row[base + 3] or 0),
                "lower_norm_distinct":       int(row[base + 4] or 0),
            }

    return stats


def _fetch_sample_values(
    conn: Any,
    quoted_table: str,
    quoted_col: str,
    sample_size: int = 100,
) -> List[str]:
    """Fetch distinct sample values for one string column."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT DISTINCT TOP ({sample_size}) {quoted_col} "
            f"FROM {quoted_table} "
            f"WHERE {quoted_col} IS NOT NULL"
        )
        return [str(r[0]) for r in cursor.fetchall()]
    except Exception:
        return []
    finally:
        cursor.close()


def analyze_string_columns(
    conn: Any,
    table_profiles: Dict[str, TableProfile],
    inventory: Dict[str, TableMetadata],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, StringProfile]]:
    """Analyze all string columns across profiled tables.

    Uses one batch stats query per table and parallel sample-value fetches,
    then processes tables in parallel.

    Args:
        conn: Open pyodbc connection (thread-safe; each worker uses its own cursor).
        table_profiles: Profiling results from profiler.profile_tables().
        inventory: Metadata inventory.
        config: Optional configuration.

    Returns:
        Nested dict: {table_key: {column_name: StringProfile}}.
    """
    cfg = config or {}
    cat_max      = cfg.get("thresholds", {}).get("string_categorical_distinct_max", 20)
    profiling_cfg = cfg.get("profiling", {})
    ts_threshold  = profiling_cfg.get("tablesample_threshold", _TABLESAMPLE_THRESHOLD)
    ts_rows       = profiling_cfg.get("tablesample_rows",      _TABLESAMPLE_ROWS)

    results: Dict[str, Dict[str, StringProfile]] = {}

    for key, t_profile in table_profiles.items():
        t_meta = inventory.get(key)
        if t_meta is None:
            continue

        string_cols = [
            col for col in t_meta.columns
            if col.data_type.split("(")[0].strip().lower() in (
                "varchar", "nvarchar", "char", "nchar", "text", "ntext"
            )
        ]
        if not string_cols:
            continue

        tablesample_rows: Optional[int] = None
        if ts_threshold > 0 and t_profile.row_count > ts_threshold:
            tablesample_rows = ts_rows

        batch_stats = _run_string_stats_batch(
            conn,
            t_meta.schema_name,
            t_meta.table_name,
            string_cols,
            tablesample_rows=tablesample_rows,
        )

        quoted_table = f"{_quote_identifier(t_meta.schema_name)}.{_quote_identifier(t_meta.table_name)}"
        sample_map: Dict[str, List[str]] = {
            col.name: _fetch_sample_values(conn, quoted_table, _quote_identifier(col.name))
            for col in string_cols
        }

        string_profiles: Dict[str, StringProfile] = {}
        for col in string_cols:
            col_profile = t_profile.columns.get(col.name)
            if col_profile is None:
                continue

            s = batch_stats.get(col.name, {})
            sample_vals = sample_map.get(col.name, [])[:20]

            avg_length = s.get("avg_length", 0.0)
            min_length = s.get("min_length", 0)
            max_length = s.get("max_length", 0)
            ws_norm    = s.get("whitespace_norm_distinct", 0)
            lower_norm = s.get("lower_norm_distinct", 0)

            normalization_reduces = (
                lower_norm < col_profile.distinct_count
                or ws_norm < col_profile.distinct_count
            )

            is_categorical = (
                col.name.lower() in _CATEGORICAL_HINT_COLUMNS
                or col_profile.distinct_count <= cat_max
            )

            is_identifier_like = False
            if sample_vals:
                matches = sum(
                    1 for p in _IDENTIFIER_PATTERNS
                    for v in sample_vals
                    if p.match(str(v))
                )
                is_identifier_like = matches >= 3

            contains_numbers = any(bool(re.search(r"\d", str(v))) for v in sample_vals)
            contains_special_chars = any(
                bool(re.search(r"[^a-zA-Z0-9\s]", str(v))) for v in sample_vals
            )

            string_profiles[col.name] = StringProfile(
                column_name=col.name,
                data_type=col.data_type,
                total_count=col_profile.row_count,
                non_null_count=col_profile.non_null_count,
                distinct_count=col_profile.distinct_count,
                null_ratio=col_profile.null_ratio,
                is_categorical=is_categorical,
                categorical_distinct_count=col_profile.distinct_count if is_categorical else 0,
                is_identifier_like=is_identifier_like,
                avg_length=round(avg_length, 2),
                min_length=min_length,
                max_length=max_length,
                contains_numbers=contains_numbers,
                contains_special_chars=contains_special_chars,
                whitespace_normalized_distinct=ws_norm,
                lower_normalized_distinct=lower_norm,
                normalization_reduces=normalization_reduces,
                sample_values=sample_vals,
            )

        if string_profiles:
            results[key] = string_profiles

    return results
