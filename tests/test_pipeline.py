"""Tests for pipeline orchestrator."""
from src.models import ColumnInfo, ColumnProfile, TableMetadata, TableProfile
from src.pipeline import run_analysis, AnalysisResult


def _make_table(schema: str, table: str, columns: list, row_count: int = 100) -> TableMetadata:
    return TableMetadata(
        schema_name=schema,
        table_name=table,
        row_count=row_count,
        columns=columns,
    )


def _make_profile(name: str, data_type: str, row_count: int, top_values: list,
                  distinct_count: int = 0) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        data_type=data_type,
        row_count=row_count,
        non_null_count=row_count,
        null_count=0,
        null_ratio=0.0,
        distinct_count=distinct_count,
        distinct_ratio=1.0,
        uniqueness_ratio=1.0,
        top_values=top_values,
    )


class TestRunAnalysis:
    def test_empty_tables(self):
        result = run_analysis({}, {})
        assert len(result.candidates) == 0
        assert len(result.scores) == 0

    def test_basic_pipeline(self):
        tables = {
            "dbo.wells": _make_table("dbo", "wells", [
                ColumnInfo(name="api", data_type="nvarchar", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="name", data_type="nvarchar", is_nullable=True, ordinal_position=2),
            ]),
            "dbo.production": _make_table("dbo", "production", [
                ColumnInfo(name="api", data_type="nvarchar", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="date", data_type="date", is_nullable=False, ordinal_position=2),
            ]),
        }
        profiles = {
            "dbo.wells": TableProfile(
                schema_name="dbo", table_name="wells", row_count=100, column_count=2,
                columns={
                    "api": _make_profile("api", "nvarchar", 100, ["A1", "A2", "A3"], distinct_count=3),
                    "name": _make_profile("name", "nvarchar", 100, ["Well1", "Well2"], distinct_count=2),
                },
                profiling_mode="A",
            ),
            "dbo.production": TableProfile(
                schema_name="dbo", table_name="production", row_count=100, column_count=2,
                columns={
                    "api": _make_profile("api", "nvarchar", 100, ["A1", "A2", "A3"], distinct_count=3),
                    "date": _make_profile("date", "date", 100, ["2024-01-01", "2024-01-02"], distinct_count=2),
                },
                profiling_mode="A",
            ),
        }

        result = run_analysis(tables, profiles)

        # Should find at least the api <-> api exact match
        assert len(result.candidates) >= 1
        api_candidates = [c for c in result.candidates if c.source_column == "api" and c.target_column == "api"]
        assert len(api_candidates) >= 1

        # Should have scores for compatible candidates
        assert len(result.scores) >= 1

        # Suggested edges should not include reject band
        edges = result.suggested_edges
        for edge in edges:
            assert edge["confidence_band"] != "reject"

    def test_type_mismatch_rejected(self):
        tables = {
            "dbo.t1": _make_table("dbo", "t1", [
                ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1),
            ]),
            "dbo.t2": _make_table("dbo", "t2", [
                ColumnInfo(name="id", data_type="date", is_nullable=False, ordinal_position=1),
            ]),
        }
        profiles = {
            "dbo.t1": TableProfile(
                schema_name="dbo", table_name="t1", row_count=100, column_count=1,
                columns={"id": _make_profile("id", "int", 100, [1, 2, 3], distinct_count=3)},
                profiling_mode="A",
            ),
            "dbo.t2": TableProfile(
                schema_name="dbo", table_name="t2", row_count=100, column_count=1,
                columns={"id": _make_profile("id", "date", 100, ["2024-01-01"], distinct_count=1)},
                profiling_mode="A",
            ),
        }

        result = run_analysis(tables, profiles)
        # id <-> id should be rejected due to type mismatch
        assert len(result.scores) == 0
        assert len(result.suggested_edges) == 0

    def test_suggested_edges_sorted(self):
        tables = {
            "dbo.t1": _make_table("dbo", "t1", [
                ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="name", data_type="nvarchar", is_nullable=True, ordinal_position=2),
            ]),
            "dbo.t2": _make_table("dbo", "t2", [
                ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="name", data_type="nvarchar", is_nullable=True, ordinal_position=2),
            ]),
        }
        profiles = {
            "dbo.t1": TableProfile(
                schema_name="dbo", table_name="t1", row_count=100, column_count=2,
                columns={
                    "id": _make_profile("id", "int", 100, [1, 2, 3], distinct_count=3),
                    "name": _make_profile("name", "nvarchar", 100, ["A", "B"], distinct_count=2),
                },
                profiling_mode="A",
            ),
            "dbo.t2": TableProfile(
                schema_name="dbo", table_name="t2", row_count=100, column_count=2,
                columns={
                    "id": _make_profile("id", "int", 100, [1, 2, 3], distinct_count=3),
                    "name": _make_profile("name", "nvarchar", 100, ["A", "B"], distinct_count=2),
                },
                profiling_mode="A",
            ),
        }

        result = run_analysis(tables, profiles)
        edges = result.suggested_edges
        confidences = [e["confidence"] for e in edges]
        assert confidences == sorted(confidences, reverse=True)


class TestAnalysisResult:
    def test_default_values(self):
        result = AnalysisResult()
        assert len(result.candidates) == 0
        assert len(result.suggested_edges) == 0