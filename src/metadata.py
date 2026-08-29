"""Extended metadata inventory.

Queries SQL Server system views for indexes, row counts, foreign keys,
and other metadata beyond basic INFORMATION_SCHEMA.
"""

from typing import Dict, List, Optional

from src.db import discover_columns, discover_foreign_keys
from src.models import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    TableMetadata,
)


def get_row_counts(conn) -> Dict[str, int]:
    """Get approximate row counts for all tables.

    Uses sys.dm_db_partition_stats for fast approximate counts.

    Args:
        conn: Open pyodbc connection.

    Returns:
        Dictionary mapping `{schema}.{table}` to row count.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            OBJECT_SCHEMA_NAME(t.object_id) AS schema_name,
            t.name AS table_name,
            SUM(p.rows) AS row_count
        FROM sys.tables t
        INNER JOIN sys.partitions p
            ON t.object_id = p.object_id
        WHERE p.index_id IN (0, 1)
        GROUP BY t.object_id, t.name
        ORDER BY schema_name, table_name
        """
    )
    counts: Dict[str, int] = {}
    for schema, table, row_count in cursor.fetchall():
        counts[f"{schema}.{table}"] = int(row_count)
    cursor.close()
    return counts


def get_indexes(conn, schema: str, table: str) -> List[IndexInfo]:
    """Get index information for a specific table.

    Args:
        conn: Open pyodbc connection.
        schema: Schema name.
        table: Table name.

    Returns:
        List of IndexInfo.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            i.name AS index_name,
            i.is_primary_key,
            i.is_unique,
            STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) AS columns
        FROM sys.indexes i
        INNER JOIN sys.index_columns ic
            ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        INNER JOIN sys.columns c
            ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        INNER JOIN sys.tables t
            ON i.object_id = t.object_id
        INNER JOIN sys.schemas s
            ON t.schema_id = s.schema_id
        WHERE s.name = ? AND t.name = ? AND i.name IS NOT NULL
        GROUP BY i.name, i.is_primary_key, i.is_unique
        ORDER BY i.is_primary_key DESC, i.name
        """,
        (schema, table),
    )
    indexes = [
        IndexInfo(
            index_name=row[0],
            is_primary_key=bool(row[1]),
            is_unique=bool(row[2]),
            column_names=row[3].split(",") if row[3] else [],
        )
        for row in cursor.fetchall()
    ]
    cursor.close()
    return indexes


def build_inventory_for_tables(
    conn,
    tables: List[tuple[str, str]],
    all_fks: Optional[List[ForeignKeyInfo]] = None,
) -> Dict[str, TableMetadata]:
    """Build metadata inventory for a specific set of tables only.

    Replaces repeated build_inventory() calls (one per selected table, each
    scanning the entire schema) with a fixed set of bulk queries regardless
    of how many tables are requested:
      - 1 query for row counts   (sys.dm_db_partition_stats)
      - 1 query for all FKs      (optional: pass pre-fetched all_fks to skip)
      - 1 query for all columns  (INFORMATION_SCHEMA.COLUMNS)
      - 1 query for all indexes  (sys.indexes)

    Args:
        conn: Open pyodbc connection.
        tables: List of (schema_name, table_name) pairs to include.
        all_fks: Pre-fetched FK list; if None, discover_foreign_keys() is called.

    Returns:
        Dictionary mapping ``{schema}.{table}`` to TableMetadata.
    """
    if not tables:
        return {}

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    unique_tables: List[tuple[str, str]] = []
    for pair in tables:
        if pair not in seen:
            seen.add(pair)
            unique_tables.append(pair)

    row_counts = get_row_counts(conn)
    if all_fks is None:
        all_fks = discover_foreign_keys(conn)

    # Bulk column fetch — one query for all requested tables.
    # Build WHERE (TABLE_SCHEMA = ? AND TABLE_NAME = ?) OR ... per pair.
    col_where = " OR ".join(
        "(TABLE_SCHEMA = ? AND TABLE_NAME = ?)" for _ in unique_tables
    )
    col_params: List[str] = [v for s, t in unique_tables for v in (s, t)]

    cursor = conn.cursor()
    cursor.execute(
        f"SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, "
        f"IS_NULLABLE, ORDINAL_POSITION "
        f"FROM INFORMATION_SCHEMA.COLUMNS "
        f"WHERE {col_where} "
        f"ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION",
        col_params,
    )
    columns_by_key: Dict[str, List[ColumnInfo]] = {}
    for row in cursor.fetchall():
        schema, table, col_name, data_type, is_nullable, ordinal = row
        key = f"{schema}.{table}"
        columns_by_key.setdefault(key, []).append(
            ColumnInfo(
                name=col_name,
                data_type=data_type,
                is_nullable=is_nullable == "YES",
                ordinal_position=ordinal,
            )
        )
    cursor.close()

    # Bulk index fetch — one query for all requested tables.
    idx_where = " OR ".join(
        "(s.name = ? AND t.name = ?)" for _ in unique_tables
    )
    idx_params: List[str] = [v for s, t in unique_tables for v in (s, t)]

    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT
            s.name AS schema_name,
            t.name AS table_name,
            i.name AS index_name,
            i.is_primary_key,
            i.is_unique,
            STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) AS cols
        FROM sys.indexes i
        INNER JOIN sys.index_columns ic
            ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        INNER JOIN sys.columns c
            ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        INNER JOIN sys.tables t
            ON i.object_id = t.object_id
        INNER JOIN sys.schemas s
            ON t.schema_id = s.schema_id
        WHERE ({idx_where}) AND i.name IS NOT NULL
        GROUP BY s.name, t.name, i.name, i.is_primary_key, i.is_unique
        ORDER BY s.name, t.name, i.is_primary_key DESC, i.name
        """,
        idx_params,
    )
    indexes_by_key: Dict[str, List[IndexInfo]] = {}
    for row in cursor.fetchall():
        schema, table, idx_name, is_pk, is_uniq, cols_str = row
        key = f"{schema}.{table}"
        indexes_by_key.setdefault(key, []).append(
            IndexInfo(
                index_name=idx_name,
                is_primary_key=bool(is_pk),
                is_unique=bool(is_uniq),
                column_names=cols_str.split(",") if cols_str else [],
            )
        )
    cursor.close()

    inventory: Dict[str, TableMetadata] = {}
    for schema, table in unique_tables:
        key = f"{schema}.{table}"
        fks = [
            fk for fk in all_fks
            if fk.source_schema == schema and fk.source_table == table
        ]
        inventory[key] = TableMetadata(
            schema_name=schema,
            table_name=table,
            row_count=row_counts.get(key, 0),
            columns=columns_by_key.get(key, []),
            indexes=indexes_by_key.get(key, []),
            foreign_keys=fks,
        )

    return inventory


def build_inventory(
    conn,
    schemas: Optional[List[str]] = None,
) -> Dict[str, TableMetadata]:
    """Build a complete metadata inventory for all (or filtered) tables.

    Args:
        conn: Open pyodbc connection.
        schemas: Optional list of schema names to restrict to. If None, all schemas.

    Returns:
        Dictionary mapping `{schema}.{table}` to TableMetadata.
    """
    row_counts = get_row_counts(conn)
    all_fks = discover_foreign_keys(conn)

    inventory: Dict[str, TableMetadata] = {}

    cursor = conn.cursor()
    cursor.execute(
        "SELECT TABLE_SCHEMA, TABLE_NAME "
        "FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_TYPE = 'BASE TABLE' "
        "ORDER BY TABLE_SCHEMA, TABLE_NAME"
    )
    for schema, table in cursor.fetchall():
        if schemas and schema not in schemas:
            continue

        key = f"{schema}.{table}"
        columns = discover_columns(conn, schema, table)
        indexes = get_indexes(conn, schema, table)
        row_count = row_counts.get(key, 0)

        # Filter FKs relevant to this table
        fks = [
            fk for fk in all_fks
            if fk.source_schema == schema and fk.source_table == table
        ]

        inventory[key] = TableMetadata(
            schema_name=schema,
            table_name=table,
            row_count=row_count,
            columns=columns,
            indexes=indexes,
            foreign_keys=fks,
        )

    cursor.close()
    return inventory