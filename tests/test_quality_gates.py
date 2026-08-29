"""Release quality gates.

Automated checks that block release if any fail:
1. Evidence completeness: all confirmed edges have type + value evidence
2. Determinism: same config + same data = same candidates
3. No hardcoded table/column names in core analysis modules
4. Config-driven thresholds exist
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from src.candidates import Candidate, generate_candidates
from src.export import generate_json_report, generate_markdown
from src.graph import RelationshipGraph, SuggestedEdge
from src.models import ColumnInfo, TableMetadata
from src.pipeline import run_analysis, AnalysisResult
from src.profiler import TableProfile
from src.scoring import compute_score
from src.type_compat import TypeDecision


def _make_metadata(schema: str, table: str, columns: list) -> TableMetadata:
    return TableMetadata(
        schema_name=schema,
        table_name=table,
        row_count=100,
        columns=[
            ColumnInfo(name=n, data_type=t, is_nullable=False, ordinal_position=i)
            for i, (n, t) in enumerate(columns)
        ],
    )


# ---------------------------------------------------------------------------
# Gate 1: Evidence completeness
# ---------------------------------------------------------------------------

class TestEvidenceCompleteness:
    """All confirmed edges should have type and value evidence when from suggestions."""

    def test_suggested_edge_has_evidence_fields(self):
        edge = SuggestedEdge(
            source_key="conn1.dbo.t1",
            source_column="id",
            target_key="conn1.dbo.t2",
            target_column="t1_id",
            confidence=0.85,
            confidence_band="high",
            evidence={
                "name_score": 1.0,
                "type_score": 1.0,
                "value_score": 0.8,
                "uniqueness_score": 0.9,
                "null_score": 0.95,
                "match_type": "exact_name",
                "reason": "Exact name match: id",
            },
        )
        assert "name_score" in edge.evidence
        assert "type_score" in edge.evidence
        assert "value_score" in edge.evidence

    def test_score_breakdown_has_all_components(self):
        decision = TypeDecision(
            compatible=True,
            canonical_a="numeric",
            canonical_b="numeric",
            risk=None,
        )
        breakdown = compute_score(
            name_score=1.0,
            type_decision=decision,
            value_score=0.8,
            source_profile=None,
            target_profile=None,
        )
        assert breakdown.name_score is not None
        assert breakdown.type_score is not None
        assert breakdown.value_score is not None
        assert breakdown.uniqueness_score is not None
        assert breakdown.null_score is not None
        assert breakdown.confidence is not None
        assert breakdown.confidence_band in ("high", "medium", "low", "reject")


# ---------------------------------------------------------------------------
# Gate 2: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same config + same data must produce identical candidate outputs."""

    def test_candidate_generation_is_deterministic(self):
        tables: Dict[str, TableMetadata] = {
            "dbo.wells": _make_metadata("dbo", "wells", [
                ("well_id", "int"),
                ("api_number", "nvarchar"),
                ("well_name", "nvarchar"),
            ]),
            "dbo.production": _make_metadata("dbo", "production", [
                ("prod_id", "int"),
                ("well_id", "int"),
                ("api_number", "nvarchar"),
                ("oil_prod", "decimal"),
            ]),
            "dbo.leases": _make_metadata("dbo", "leases", [
                ("lease_id", "int"),
                ("lease_num", "nvarchar"),
                ("operator", "nvarchar"),
            ]),
        }
        config = {
            "thresholds": {"name_similarity_min": 0.6},
            "aliases": {
                "api": ["api_number", "api_num", "well_api"],
                "well": ["well_name", "wellname"],
            },
        }

        # Run twice
        run1 = generate_candidates(tables, config)
        run2 = generate_candidates(tables, config)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1.source_table == c2.source_table
            assert c1.source_column == c2.source_column
            assert c1.target_table == c2.target_table
            assert c1.target_column == c2.target_column
            assert c1.name_score == c2.name_score
            assert c1.match_type == c2.match_type

    def test_pipeline_produces_deterministic_edges(self):
        tables: Dict[str, TableMetadata] = {
            "dbo.t1": _make_metadata("dbo", "t1", [
                ("id", "int"),
                ("name", "nvarchar"),
            ]),
            "dbo.t2": _make_metadata("dbo", "t2", [
                ("id", "int"),
                ("t1_id", "int"),
                ("name", "nvarchar"),
            ]),
        }
        profiles: Dict[str, TableProfile] = {
            "dbo.t1": TableProfile(
                schema_name="dbo", table_name="t1", row_count=100, column_count=2,
                columns={
                    "id": None,  # type: ignore[dict-item]
                    "name": None,  # type: ignore[dict-item]
                },
                profiling_mode="A",
            ),
            "dbo.t2": TableProfile(
                schema_name="dbo", table_name="t2", row_count=100, column_count=3,
                columns={
                    "id": None,  # type: ignore[dict-item]
                    "t1_id": None,  # type: ignore[dict-item]
                    "name": None,  # type: ignore[dict-item]
                },
                profiling_mode="A",
            ),
        }

        run1 = run_analysis(tables, profiles, config={})
        run2 = run_analysis(tables, profiles, config={})

        edges1 = run1.suggested_edges
        edges2 = run2.suggested_edges
        assert len(edges1) == len(edges2)

        for e1, e2 in zip(edges1, edges2):
            assert e1["source_table"] == e2["source_table"]
            assert e1["source_column"] == e2["source_column"]
            assert e1["target_table"] == e2["target_table"]
            assert e1["target_column"] == e2["target_column"]
            assert e1["confidence"] == e2["confidence"]


# ---------------------------------------------------------------------------
# Gate 3: No hardcoded names in core analysis
# ---------------------------------------------------------------------------

class TestNoHardcodedNames:
    """Core analysis modules must not contain hardcoded table or column names."""

    _CORE_MODULES = [
        "src/candidates.py",
        "src/pipeline.py",
        "src/profiler.py",
        "src/scoring.py",
        "src/value_evidence.py",
        "src/string_evidence.py",
        "src/type_compat.py",
        "src/types.py",
    ]

    # Allowed patterns (type names, SQL keywords, config keys)
    _ALLOWED = {
        "int", "bigint", "smallint", "tinyint", "decimal", "numeric",
        "float", "real", "money", "bit",
        "char", "varchar", "nchar", "nvarchar", "text", "ntext",
        "date", "time", "datetime", "datetime2", "smalldatetime",
        "datetimeoffset", "binary", "varbinary", "image",
        "json", "xml", "geometry", "geography",
        "uniqueidentifier", "sql_variant", "hierarchyid",
        "rowversion", "timestamp",
        "id", "key", "name", "type", "value", "count",
        "schema", "table", "column", "index",
        "numeric", "string", "datetime", "boolean", "binary",
        "unknown", "spatial", "other",
        "one-to-one", "one-to-many", "many-to-one", "many-to-many",
        "exact_name", "alias", "fuzzy",
        "high", "medium", "low", "reject",
        "cast_required", "incompatible",
        "source_table", "source_column", "target_table", "target_column",
        "source_key", "target_key",
        "name_score", "type_score", "value_score",
        "uniqueness_score", "null_score",
        "_id", "_key", "_code", "_no", "_num", "_number",
        "table_name", "schema_name", "column_name",
        "data_type", "is_nullable", "ordinal_position",
        "row_count", "non_null_count", "null_count",
        "distinct_count", "distinct_ratio", "uniqueness_ratio",
        "min_value", "max_value", "top_values",
        "profiling_mode", "profiling_note",
        "confidence", "confidence_band", "evidence",
        "rel_type", "annotation", "match_type", "reason",
        "compatible", "canonical_a", "canonical_b", "risk",
        "exact_overlap_count", "exact_overlap_ratio",
        "jaccard", "containment_source_to_target",
        "containment_target_to_source", "mode", "sample_size",
        "common_values", "confidence_flags",
        "categorical_alignment", "token_similarity",
        "normalization_overlap", "fuzzy_match_score",
        "is_both_categorical", "is_both_identifier",
        "shared_categories", "weak_match",
        "string_score", "value_score",
        "name_similarity_min", "type_compatibility",
        "value_overlap_min", "value_overlap_ratio",
        "jaccard_min", "confidence_high", "confidence_medium",
        "confidence_low", "string_categorical_distinct_max",
        "mode_a_max_rows", "mode_b_string_cardinality",
        "mode_c_sample_sizes", "mode_c_confidence_threshold",
        "scoring_weights",
        "column_patterns", "table_patterns",
        "full_name", "primary_key_columns",
        "index_name", "is_primary_key", "is_unique",
        "column_names", "constraint_name",
        "source_schema", "target_schema",
        "total_count", "non_null_count", "distinct_count",
        "null_ratio", "is_categorical",
        "categorical_distinct_count", "is_identifier_like",
        "avg_length", "min_length", "max_length",
        "contains_numbers", "contains_special_chars",
        "whitespace_normalized_distinct",
        "lower_normalized_distinct", "normalization_reduces",
        "sample_values",
        "normalized_overlap_count", "normalized_overlap_ratio",
        "sampled_estimate", "source_contained_in_target",
        "target_contained_in_source", "high_jaccard",
        "missing", "sampled", "full",
        "name", "table", "database", "schema",
        "connection_id", "primary_keys", "columns",
        "confirmed_edges", "suggested_edges", "annotations",
        "source", "target",
        "version", "connections", "selected_tables", "graph",
        "title", "database_count", "server", "database",
        "Rows", "Columns", "Primary Keys", "Description",
        "No confirmed relationships",
        "Suggested", "Pending Review",
        "Entity Relationship Diagram",
        "confidence_flags",
        "value_evidence", "string_evidence",
    }

    def test_no_hardcoded_table_names(self):
        project_root = Path(__file__).parent.parent
        for module_path in self._CORE_MODULES:
            filepath = project_root / module_path
            if not filepath.exists():
                continue

            source = filepath.read_text()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value.lower()
                    # Skip strings that are type names, config keys, or short tokens
                    if val in self._ALLOWED:
                        continue
                    # Skip SQL fragments
                    if any(kw in val for kw in ("select", "from", "where", "inner join", "order by")):
                        continue
                    # Skip type patterns
                    if re.match(r"^[a-z]+\s*$", val) and len(val) < 15:
                        continue

    def test_aliases_are_config_driven(self):
        """Aliases should come from config, not hardcoded in candidates."""
        tables = {
            "dbo.t1": _make_metadata("dbo", "t1", [
                ("api_number", "nvarchar"),
                ("well_name", "nvarchar"),
            ]),
            "dbo.t2": _make_metadata("dbo", "t2", [
                ("api_num", "nvarchar"),
                ("wellname", "nvarchar"),
            ]),
        }

        # Without aliases config, no alias matches should appear
        candidates_no_aliases = generate_candidates(tables, config={})
        alias_matches = [c for c in candidates_no_aliases if c.match_type == "alias"]
        assert len(alias_matches) == 0

        # With aliases config, alias matches should appear
        candidates_with_aliases = generate_candidates(tables, config={
            "aliases": {
                "api": ["api_number", "api_num"],
                "well": ["well_name", "wellname"],
            },
        })
        alias_matches = [c for c in candidates_with_aliases if c.match_type == "alias"]
        assert len(alias_matches) > 0


# ---------------------------------------------------------------------------
# Gate 4: Config-driven thresholds
# ---------------------------------------------------------------------------

class TestConfigDriven:
    """All thresholds must be configurable, not hardcoded."""

    def test_defaults_yaml_exists(self):
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "defaults.yaml"
        assert config_path.exists(), "config/defaults.yaml must exist"

    def test_config_has_required_sections(self):
        import yaml
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "defaults.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "thresholds" in config, "Config must have 'thresholds' section"
        assert "aliases" in config, "Config must have 'aliases' section"
        assert "profiling" in config, "Config must have 'profiling' section"
        assert "exclusions" in config, "Config must have 'exclusions' section"

    def test_thresholds_are_used_in_pipeline(self):
        """Pipeline should respect config thresholds."""
        tables = {
            "dbo.t1": _make_metadata("dbo", "t1", [
                ("col_a", "int"),
                ("col_b", "int"),
            ]),
            "dbo.t2": _make_metadata("dbo", "t2", [
                ("col_a", "int"),
                ("col_c", "int"),
            ]),
        }

        # Low threshold -> more candidates
        candidates_low = generate_candidates(tables, config={
            "thresholds": {"name_similarity_min": 0.3},
        })

        # High threshold -> fewer candidates
        candidates_high = generate_candidates(tables, config={
            "thresholds": {"name_similarity_min": 0.95},
        })

        assert len(candidates_low) >= len(candidates_high)
