"""Regression tests for validate_sql_safety (Chat read-only SQL guard).

The guard was previously a naive keyword-substring check that was bypassed by
separating a write/DDL statement with a semicolon, a newline, or a SQL comment.
These tests pin the parser-based behaviour: reject everything that is not a
single, read-only SELECT/WITH statement.
"""

import pytest

from src.chat.context_builder import validate_sql_safety

# Payloads that must be blocked (the old guard let all of these through).
MUST_BLOCK = [
    "SELECT 1; DROP TABLE x",
    "SELECT 1;DROP TABLE x",
    "SELECT 1 -- comment\nDROP TABLE x",
    "SELECT 1/**/DROP TABLE x",
    "SELECT 1;\nDELETE FROM x",
    "SELECT 1\tUPDATE x SET a = 1",
    "SELECT 1 INTO newtable",              # SELECT ... INTO is a write
    "WITH t AS (SELECT 1 AS a) SELECT 1; DROP TABLE x",
    "DROP TABLE x",
    "DELETE FROM x",
    "EXEC xp_cmdshell 'dir'",
    "",
    "   ;   ",
]

# Legitimate read-only queries that must pass unchanged.
MUST_ALLOW = [
    "SELECT TOP (10) a FROM b WHERE c = 1",
    "SELECT a, b FROM t1 JOIN t2 ON t1.id = t2.id",
    "WITH t AS (SELECT 1 AS a) SELECT a FROM t",
    "SELECT a FROM b UNION SELECT c FROM d",
    "SELECT x FROM (SELECT 1 AS x) s",
    "SELECT department, COUNT(*) AS n FROM emp GROUP BY department ORDER BY n DESC",
]


@pytest.mark.parametrize("sql", MUST_BLOCK)
def test_unsafe_or_bypass_sql_is_blocked(sql: str) -> None:
    assert validate_sql_safety(sql) is not None


@pytest.mark.parametrize("sql", MUST_ALLOW)
def test_read_only_queries_are_allowed(sql: str) -> None:
    assert validate_sql_safety(sql) is None
