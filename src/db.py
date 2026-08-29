"""SQL Server connection and schema discovery.

Provides pyodbc-based connection management and INFORMATION_SCHEMA queries
for discovering tables, columns, types, and nullability.
"""

from typing import Dict, List, Optional

from src.models import ColumnInfo, ForeignKeyInfo, TableInfo


def connect_to_sql_server(
    server: str,
    database: str,
    username: str,
    password: str,
    driver: str = "ODBC Driver 18 for SQL Server",
    timeout: int = 10,
    trust_server_certificate: bool = True,
) -> "pyodbc.Connection":
    """Connect to a SQL Server instance.

    Args:
        server: Server hostname or IP.
        database: Database name.
        username: SQL Server login name.
        password: SQL Server login password.
        driver: ODBC driver name.
        timeout: Connection timeout in seconds.
        trust_server_certificate: Whether to trust server certificate during TLS.

    Returns:
        An open pyodbc.Connection.

    Raises:
        pyodbc.Error: If connection fails.
    """
    import pyodbc

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Connection Timeout={timeout};"
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
    )
    return pyodbc.connect(conn_str)


def discover_tables(conn: "pyodbc.Connection") -> Dict[str, TableInfo]:
    """Discover all user tables in the database.

    Args:
        conn: An open pyodbc connection.

    Returns:
        Dictionary mapping `{schema}.{table}` to TableInfo.
    """
    cursor = conn.cursor()
    tables: Dict[str, TableInfo] = {}
    cursor.execute(
        "SELECT TABLE_SCHEMA, TABLE_NAME "
        "FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_TYPE = 'BASE TABLE' "
        "ORDER BY TABLE_SCHEMA, TABLE_NAME"
    )
    for schema, table in cursor.fetchall():
        key = f"{schema}.{table}"
        tables[key] = TableInfo(schema_name=schema, table_name=table)
    cursor.close()
    return tables


def discover_columns(
    conn: "pyodbc.Connection", schema: str, table: str
) -> List[ColumnInfo]:
    """Discover columns for a specific table.

    Args:
        conn: An open pyodbc connection.
        schema: Schema name.
        table: Table name.

    Returns:
        List of ColumnInfo in ordinal position order.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION "
        "FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
        "ORDER BY ORDINAL_POSITION",
        (schema, table),
    )
    columns = [
        ColumnInfo(
            name=row[0],
            data_type=row[1],
            is_nullable=row[2] == "YES",
            ordinal_position=row[3],
        )
        for row in cursor.fetchall()
    ]
    cursor.close()
    return columns


def discover_foreign_keys(conn: "pyodbc.Connection") -> List[ForeignKeyInfo]:
    """Discover all foreign key relationships in the database.

    Args:
        conn: An open pyodbc connection.

    Returns:
        List of ForeignKeyInfo.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            OBJECT_SCHEMA_NAME(fk.parent_object_id) AS source_schema,
            OBJECT_NAME(fk.parent_object_id) AS source_table,
            pc.name AS source_column,
            OBJECT_SCHEMA_NAME(fk.referenced_object_id) AS target_schema,
            OBJECT_NAME(fk.referenced_object_id) AS target_table,
            rc.name AS target_column,
            fk.name AS constraint_name
        FROM sys.foreign_keys fk
        INNER JOIN sys.foreign_key_columns fkc
            ON fk.object_id = fkc.constraint_object_id
        INNER JOIN sys.columns pc
            ON fkc.parent_object_id = pc.object_id
            AND fkc.parent_column_id = pc.column_id
        INNER JOIN sys.columns rc
            ON fkc.referenced_object_id = rc.object_id
            AND fkc.referenced_column_id = rc.column_id
        ORDER BY source_schema, source_table, constraint_name
        """
    )
    fks = [
        ForeignKeyInfo(
            source_schema=row[0],
            source_table=row[1],
            source_column=row[2],
            target_schema=row[3],
            target_table=row[4],
            target_column=row[5],
            constraint_name=row[6],
        )
        for row in cursor.fetchall()
    ]
    cursor.close()
    return fks


def close_connection(conn: Optional["pyodbc.Connection"]) -> None:
    """Safely close a database connection.

    Args:
        conn: Connection to close, or None.
    """
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def discover_databases(
    server: str,
    username: str = "",
    password: str = "",
    driver: str = "ODBC Driver 18 for SQL Server",
    timeout: int = 10,
    trust_server_certificate: bool = True,
) -> List[str]:
    """Discover user databases available on a SQL Server instance.

    Connects to the server-level (master database) and queries sys.databases.

    Args:
        server: Server hostname or IP.
        username: SQL Server login name (empty for Windows auth).
        password: SQL Server login password.
        driver: ODBC driver name.
        timeout: Connection timeout in seconds.
        trust_server_certificate: Whether to trust server certificate during TLS.

    Returns:
        List of user database names (excludes system databases).
    """
    import pyodbc

    system_databases = {"master", "tempdb", "model", "msdb"}

    if username:
        conn_str = (
            f"DRIVER={{{driver}}}"
            f"SERVER={server};"
            f"DATABASE=master;"
            f"UID={username};"
            f"PWD={password};"
            f"Connection Timeout={timeout};"
            f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}}"
            f"SERVER={server};"
            f"DATABASE=master;"
            f"Trusted_Connection=yes;"
            f"Connection Timeout={timeout};"
            f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
        )

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sys.databases "
        "WHERE database_id > 4 AND state = 0 "
        "ORDER BY name"
    )
    databases = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    # Filter out any system databases that slipped through
    return [db for db in databases if db not in system_databases]
