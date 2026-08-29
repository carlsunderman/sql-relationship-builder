"""SQL Executor agent: runs a generated SELECT against pyodbc safely."""

import logging
import time
from typing import Any, Optional

from src.chat.context_builder import validate_sql_safety
from src.chat.models import QueryResult, QueryStatus

logger = logging.getLogger(__name__)

DEFAULT_ROW_LIMIT = 1000
DEFAULT_TIMEOUT = 30


class SQLExecutor:
    """Executes SQL queries against a live pyodbc connection with safety guards."""

    def __init__(
        self,
        row_limit: int = DEFAULT_ROW_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.row_limit = row_limit
        self.timeout = timeout

    def execute(self, sql: str, connection: Any) -> QueryResult:
        """Run the query and return a structured result.

        Args:
            sql: The T-SQL SELECT statement to execute.
            connection: An open pyodbc.Connection.

        Returns:
            QueryResult with status, columns, rows, and any error info.
        """
        if connection is None:
            return QueryResult(
                status=QueryStatus.ERROR,
                error_message="No database connection available",
            )

        safety_error = validate_sql_safety(sql)
        if safety_error:
            return QueryResult(
                status=QueryStatus.BLOCKED,
                error_message=safety_error,
            )

        # Inject row limit if not already present
        bounded_sql = self._apply_row_limit(sql)

        start = time.time()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(bounded_sql)
            columns = [col[0] for col in cursor.description] if cursor.description else []

            rows = cursor.fetchmany(self.row_limit + 1)
            truncated = len(rows) > self.row_limit
            if truncated:
                rows = rows[: self.row_limit]

            # Convert non-serializable types
            rows_out = [
                [self._serialize_value(v) for v in row]
                for row in rows
            ]

            elapsed = (time.time() - start) * 1000.0
            return QueryResult(
                status=QueryStatus.SUCCESS,
                columns=columns,
                rows=rows_out,
                row_count=len(rows_out),
                truncated=truncated,
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000.0
            logger.error("Query execution failed: %s", e)
            return QueryResult(
                status=QueryStatus.ERROR,
                error_message=str(e),
                execution_time_ms=elapsed,
            )
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass

    def _apply_row_limit(self, sql: str) -> str:
        """Add TOP (N) if the query doesn't already have one."""
        upper = sql.upper().lstrip()
        if upper.startswith("SELECT") and " TOP " not in upper:
            # Insert TOP after SELECT
            return sql.replace("SELECT", f"SELECT TOP ({self.row_limit})", 1)
        return sql

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Convert pyodbc values to JSON-serializable types."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        # datetime, date, Decimal, bytes, etc.
        return str(value)
