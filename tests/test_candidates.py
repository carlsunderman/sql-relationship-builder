"""Tests for candidate generation."""
from src.candidates import generate_candidates, _name_similarity, _normalize_name
from src.models import ColumnInfo, TableMetadata


def _make_table(schema: str, table: str, columns: list) -> TableMetadata:
    return TableMetadata(
        schema_name=schema,
        table_name=table,
        row_count=100,
        columns=columns,
    )


class TestNormalizeName:
    def test_strips_id_suffix(self):
        assert _normalize_name("well_id") == "well"

    def test_strips_key_suffix(self):
        assert _normalize_name("lease_key") == "lease"

    def test_strips_code_suffix(self):
        assert _normalize_name("area_code") == "area"

    def test_no_suffix(self):
        assert _normalize_name("operator_name") == "operator_name"


class TestNameSimilarity:
    def test_exact_match(self):
        assert _name_similarity("api", "api") >= 0.99

    def test_suffix_variation(self):
        # well_id vs well should score high (well_id normalizes to well)
        sim = _name_similarity("well_id", "well")
        assert sim >= 0.9

    def test_different_names(self):
        sim = _name_similarity("operator", "lease")
        assert sim < 0.5


class TestGenerateCandidates:
    def test_exact_name_match(self):
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
        candidates = generate_candidates(tables)
        # Should find api <-> api exact match
        api_candidates = [c for c in candidates if c.source_column == "api" and c.target_column == "api"]
        assert len(api_candidates) == 1
        assert api_candidates[0].match_type == "exact_name"
        assert api_candidates[0].name_score == 1.0

    def test_alias_match(self):
        tables = {
            "dbo.wells": _make_table("dbo", "wells", [
                ColumnInfo(name="api_number", data_type="nvarchar", is_nullable=False, ordinal_position=1),
            ]),
            "dbo.production": _make_table("dbo", "production", [
                ColumnInfo(name="well_api", data_type="nvarchar", is_nullable=False, ordinal_position=1),
            ]),
        }
        config = {
            "aliases": {
                "api": ["api_number", "well_api"],
            },
        }
        candidates = generate_candidates(tables, config)
        alias_candidates = [c for c in candidates if c.match_type == "alias"]
        assert len(alias_candidates) == 1
        assert alias_candidates[0].name_score == 0.85

    def test_fuzzy_match(self):
        tables = {
            "dbo.wells": _make_table("dbo", "wells", [
                ColumnInfo(name="well_name", data_type="nvarchar", is_nullable=False, ordinal_position=1),
            ]),
            "dbo.production": _make_table("dbo", "production", [
                ColumnInfo(name="wellname", data_type="nvarchar", is_nullable=False, ordinal_position=1),
            ]),
        }
        config = {"thresholds": {"name_similarity_min": 0.6}}
        candidates = generate_candidates(tables, config)
        fuzzy_candidates = [c for c in candidates if c.match_type == "fuzzy"]
        assert len(fuzzy_candidates) == 1
        assert fuzzy_candidates[0].name_score >= 0.6

    def test_type_mismatch_filtered(self):
        tables = {
            "dbo.t1": _make_table("dbo", "t1", [
                ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1),
            ]),
            "dbo.t2": _make_table("dbo", "t2", [
                ColumnInfo(name="id", data_type="date", is_nullable=False, ordinal_position=1),
            ]),
        }
        candidates = generate_candidates(tables)
        # id <-> id should not appear because types differ
        assert not any(c.source_column == "id" for c in candidates)

    def test_no_candidates_empty_tables(self):
        tables = {
            "dbo.t1": _make_table("dbo", "t1", []),
            "dbo.t2": _make_table("dbo", "t2", []),
        }
        candidates = generate_candidates(tables)
        assert len(candidates) == 0

    def test_sorted_by_name_score(self):
        tables = {
            "dbo.t1": _make_table("dbo", "t1", [
                ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="name", data_type="nvarchar", is_nullable=True, ordinal_position=2),
            ]),
            "dbo.t2": _make_table("dbo", "t2", [
                ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="label", data_type="nvarchar", is_nullable=True, ordinal_position=2),
            ]),
        }
        candidates = generate_candidates(tables)
        scores = [c.name_score for c in candidates]
        assert scores == sorted(scores, reverse=True)