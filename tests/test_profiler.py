"""Tests for profiler module (no pyodbc needed)."""
from unittest.mock import Mock, call

from src.profiler import select_profiling_mode, _run_column_profiles_pushdown
from src.models import ColumnInfo, ColumnProfile, TableProfile


class TestSelectProfilingMode:
    def test_mode_a_small_table(self):
        assert select_profiling_mode(row_count=1000, string_columns=5, mode_a_max_rows=100000) == "A"

    def test_mode_a_at_threshold(self):
        assert select_profiling_mode(row_count=100000, string_columns=5, mode_a_max_rows=100000) == "A"

    def test_mode_b_large_table_few_strings(self):
        mode = select_profiling_mode(
            row_count=200000, string_columns=10,
            mode_a_max_rows=100000, mode_b_string_cardinality=5000,
        )
        assert mode == "B"

    def test_mode_c_large_table_many_strings(self):
        mode = select_profiling_mode(
            row_count=200000, string_columns=30,
            mode_a_max_rows=100000, mode_b_string_cardinality=5000,
        )
        assert mode == "C"


class TestPushdownProfiler:
    def test_bit_column_does_not_use_min_max(self):
        """Bit columns must not appear in MIN()/MAX() expressions (SQL Server error 8117)."""
        captured_queries = []

        cursor = Mock()
        cursor.fetchone.return_value = (
            10,   # COUNT_BIG(*) — row_count
            10,   # COUNT_BIG([is_active]) — non_null
            0,    # COUNT_BIG(CASE WHEN ...) — null_count
            2,    # COUNT(DISTINCT [is_active]) — distinct
            None, # NULL placeholder — min
            None, # NULL placeholder — max
        )
        cursor.fetchall.return_value = [(1, 7), (0, 3)]

        def _capture_execute(query, params=None):
            captured_queries.append(query)

        cursor.execute.side_effect = _capture_execute

        conn = Mock()
        conn.cursor.return_value = cursor

        cols = [ColumnInfo(name="is_active", data_type="bit", is_nullable=False, ordinal_position=1)]
        profiles = _run_column_profiles_pushdown(conn, "dbo", "users", cols)

        # Verify no query applied MIN/MAX directly to the bit column.
        for q in captured_queries:
            assert "MIN([is_active])" not in q, f"MIN applied to bit column in: {q}"
            assert "MAX([is_active])" not in q, f"MAX applied to bit column in: {q}"

        p = profiles["is_active"]
        assert p.row_count == 10
        assert p.distinct_count == 2
        assert p.min_value is None
        assert p.max_value is None

    def test_numeric_column_includes_min_max(self):
        """Numeric columns should have min/max populated from the stats query."""
        cursor = Mock()
        cursor.fetchone.return_value = (
            1000, # row_count
            950,  # non_null
            50,   # null_count
            200,  # distinct
            1,    # min
            999,  # max
        )
        cursor.fetchall.return_value = [(42, 100), (7, 80)]

        conn = Mock()
        conn.cursor.return_value = cursor

        cols = [ColumnInfo(name="score", data_type="int", is_nullable=True, ordinal_position=1)]
        profiles = _run_column_profiles_pushdown(conn, "dbo", "results", cols)

        p = profiles["score"]
        assert p.row_count == 1000
        assert p.non_null_count == 950
        assert p.distinct_count == 200
        assert p.min_value == 1
        assert p.max_value == 999
        assert p.top_values == [42, 7]


class TestDataClasses:
    def test_column_profile_defaults(self):
        cp = ColumnProfile(
            name="id", data_type="int",
            row_count=100, non_null_count=100, null_count=0,
            null_ratio=0.0, distinct_count=100, distinct_ratio=1.0,
            uniqueness_ratio=1.0,
        )
        assert cp.profiling_mode == "A"
        assert cp.top_values == []

    def test_table_profile(self):
        cp = ColumnProfile(
            name="id", data_type="int",
            row_count=100, non_null_count=100, null_count=0,
            null_ratio=0.0, distinct_count=100, distinct_ratio=1.0,
            uniqueness_ratio=1.0,
        )
        tp = TableProfile(
            schema_name="dbo", table_name="t", row_count=100,
            column_count=1, columns={"id": cp},
            profiling_mode="A",
        )
        assert tp.schema_name == "dbo"
        assert tp.columns["id"].distinct_count == 100
