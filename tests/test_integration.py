"""Integration tests for the full pipeline on a fixed database snapshot.

Exercises: metadata -> profiling -> analysis -> graph -> curation -> export
-> drift -> save/load, all with a realistic multi-table snapshot.
"""

import json
import os
import tempfile
from typing import Any, Dict, List

from src.candidates import generate_candidates
from src.drift import detect_drift, save_snapshot, load_snapshot
from src.export import generate_json_report, generate_markdown, write_json_report
from src.graph import RelationshipGraph, SuggestedEdge
from src.metadata import build_inventory
from src.models import (
    ColumnInfo,
    ColumnProfile,
    IndexInfo,
    StringProfile,
    TableMetadata,
    TableProfile,
)
from src.pipeline import run_analysis, AnalysisResult
from src.state import build_save_data, load_relationships, save_relationships
from src.string_evidence import compute_string_evidence
from src.types import canonicalize_type, CanonicalType


# ---------------------------------------------------------------------------
# Fixed database snapshot fixture: Wells + Production + Leases domain
# ---------------------------------------------------------------------------

def _build_snapshot() -> Dict[str, TableMetadata]:
    """Build a realistic metadata snapshot for the wells/production domain."""
    return {
        "dbo.wells": TableMetadata(
            schema_name="dbo",
            table_name="wells",
            row_count=5000,
            columns=[
                ColumnInfo(name="well_id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="api_number", data_type="nvarchar(20)", is_nullable=False, ordinal_position=2),
                ColumnInfo(name="well_name", data_type="nvarchar(100)", is_nullable=True, ordinal_position=3),
                ColumnInfo(name="operator", data_type="nvarchar(100)", is_nullable=True, ordinal_position=4),
                ColumnInfo(name="status", data_type="nvarchar(20)", is_nullable=True, ordinal_position=5),
                ColumnInfo(name="lease_num", data_type="nvarchar(20)", is_nullable=True, ordinal_position=6),
                ColumnInfo(name="block_num", data_type="nvarchar(20)", is_nullable=True, ordinal_position=7),
                ColumnInfo(name="spud_date", data_type="datetime2", is_nullable=True, ordinal_position=8),
            ],
            indexes=[
                IndexInfo(index_name="PK_wells", is_primary_key=True, is_unique=True, column_names=["well_id"]),
            ],
        ),
        "dbo.production": TableMetadata(
            schema_name="dbo",
            table_name="production",
            row_count=250000,
            columns=[
                ColumnInfo(name="prod_id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="well_id", data_type="int", is_nullable=False, ordinal_position=2),
                ColumnInfo(name="api_number", data_type="nvarchar(20)", is_nullable=True, ordinal_position=3),
                ColumnInfo(name="prod_date", data_type="date", is_nullable=False, ordinal_position=4),
                ColumnInfo(name="oil_prod", data_type="decimal(18,2)", is_nullable=True, ordinal_position=5),
                ColumnInfo(name="gas_prod", data_type="decimal(18,2)", is_nullable=True, ordinal_position=6),
                ColumnInfo(name="water_prod", data_type="decimal(18,2)", is_nullable=True, ordinal_position=7),
                ColumnInfo(name="status", data_type="nvarchar(20)", is_nullable=True, ordinal_position=8),
            ],
            indexes=[
                IndexInfo(index_name="PK_production", is_primary_key=True, is_unique=True, column_names=["prod_id"]),
            ],
        ),
        "dbo.leases": TableMetadata(
            schema_name="dbo",
            table_name="leases",
            row_count=1200,
            columns=[
                ColumnInfo(name="lease_id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="lease_num", data_type="nvarchar(20)", is_nullable=False, ordinal_position=2),
                ColumnInfo(name="operator", data_type="nvarchar(100)", is_nullable=True, ordinal_position=3),
                ColumnInfo(name="block_num", data_type="nvarchar(20)", is_nullable=True, ordinal_position=4),
                ColumnInfo(name="area_code", data_type="nvarchar(10)", is_nullable=True, ordinal_position=5),
                ColumnInfo(name="eff_date", data_type="date", is_nullable=True, ordinal_position=6),
            ],
            indexes=[
                IndexInfo(index_name="PK_leases", is_primary_key=True, is_unique=True, column_names=["lease_id"]),
            ],
        ),
        "dbo.operators": TableMetadata(
            schema_name="dbo",
            table_name="operators",
            row_count=350,
            columns=[
                ColumnInfo(name="operator_id", data_type="int", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="company_name", data_type="nvarchar(200)", is_nullable=False, ordinal_position=2),
                ColumnInfo(name="operator_name", data_type="nvarchar(100)", is_nullable=True, ordinal_position=3),
                ColumnInfo(name="status", data_type="nvarchar(20)", is_nullable=True, ordinal_position=4),
            ],
            indexes=[
                IndexInfo(index_name="PK_operators", is_primary_key=True, is_unique=True, column_names=["operator_id"]),
            ],
        ),
    }


def _build_profiles(
    inventory: Dict[str, TableMetadata],
) -> Dict[str, TableProfile]:
    """Build realistic column profiles for the snapshot."""
    profiles: Dict[str, TableProfile] = {}

    for key, meta in inventory.items():
        col_profiles: Dict[str, ColumnProfile] = {}

        for col in meta.columns:
            # Simulate realistic profile stats
            row_count = meta.row_count
            is_pk = col.name in meta.primary_key_columns
            is_fk_like = col.name.endswith("_id") or col.name in ("well_id", "lease_num", "api_number")

            if is_pk:
                distinct_count = row_count
                null_count = 0
            elif is_fk_like:
                # FK-like columns: high distinct, low nulls
                distinct_count = min(row_count, 5000)
                null_count = int(row_count * 0.02)
            elif col.name in ("status",):
                # Categorical: few distinct values
                distinct_count = 5
                null_count = int(row_count * 0.05)
            elif col.name in ("operator", "operator_name", "company_name"):
                distinct_count = 350
                null_count = int(row_count * 0.03)
            elif col.name in ("well_name",):
                distinct_count = row_count
                null_count = int(row_count * 0.01)
            else:
                distinct_count = max(100, row_count // 10)
                null_count = int(row_count * 0.05)

            non_null = row_count - null_count
            null_ratio = null_count / row_count if row_count else 0.0
            distinct_ratio = distinct_count / non_null if non_null else 0.0
            uniqueness_ratio = distinct_count / row_count if row_count else 0.0

            # Simulate top values for overlap detection
            top_values: List[Any] = []
            if col.name == "well_id":
                top_values = list(range(1, 11))
            elif col.name == "api_number":
                top_values = [f"430-{i:05d}" for i in range(1, 11)]
            elif col.name == "lease_num":
                top_values = [f"L-{i:04d}" for i in range(1, 11)]
            elif col.name == "block_num":
                top_values = [f"B{i:03d}" for i in range(1, 11)]
            elif col.name == "status":
                top_values = ["Active", "Inactive", "Suspended", "Plugged", "Pending"]
            elif col.name == "operator":
                top_values = ["Operator A", "Operator B", "Operator C"]
            elif col.name == "area_code":
                top_values = ["TX", "OK", "LA", "CO", "WY"]

            col_profiles[col.name] = ColumnProfile(
                name=col.name,
                data_type=col.data_type,
                row_count=row_count,
                non_null_count=non_null,
                null_count=null_count,
                null_ratio=round(null_ratio, 6),
                distinct_count=distinct_count,
                distinct_ratio=round(distinct_ratio, 6),
                uniqueness_ratio=round(uniqueness_ratio, 6),
                top_values=top_values,
                profiling_mode="A",
            )

        profiles[key] = TableProfile(
            schema_name=meta.schema_name,
            table_name=meta.table_name,
            row_count=meta.row_count,
            column_count=len(meta.columns),
            columns=col_profiles,
            profiling_mode="A",
            profiling_note="Full pushdown profiling",
        )

    return profiles


def _build_string_profiles(
    inventory: Dict[str, TableMetadata],
    table_profiles: Dict[str, TableProfile],
) -> Dict[str, Dict[str, StringProfile]]:
    """Build realistic string profiles for string columns."""
    results: Dict[str, Dict[str, StringProfile]] = {}

    for key, meta in inventory.items():
        t_profile = table_profiles.get(key)
        if t_profile is None:
            continue

        string_profiles: Dict[str, StringProfile] = {}

        for col in meta.columns:
            base = col.data_type.split("(")[0].strip().lower()
            if base not in ("varchar", "nvarchar", "char", "nchar", "text", "ntext"):
                continue

            cp = t_profile.columns.get(col.name)
            if cp is None:
                continue

            is_categorical = cp.distinct_count <= 20
            sample_values = cp.top_values if cp.top_values else []

            avg_len = 10.0
            min_len = 1
            max_len = 50
            if col.name == "api_number":
                avg_len, min_len, max_len = 9.0, 7, 12
            elif col.name == "status":
                avg_len, min_len, max_len = 7.0, 4, 10
            elif col.name == "well_name":
                avg_len, min_len, max_len = 15.0, 3, 50

            string_profiles[col.name] = StringProfile(
                column_name=col.name,
                data_type=col.data_type,
                total_count=cp.row_count,
                non_null_count=cp.non_null_count,
                distinct_count=cp.distinct_count,
                null_ratio=cp.null_ratio,
                is_categorical=is_categorical,
                categorical_distinct_count=cp.distinct_count if is_categorical else 0,
                is_identifier_like=col.name in ("api_number", "lease_num", "block_num"),
                avg_length=avg_len,
                min_length=min_len,
                max_length=max_len,
                contains_numbers=col.name in ("api_number", "lease_num", "block_num"),
                contains_special_chars=col.name == "api_number",
                whitespace_normalized_distinct=cp.distinct_count,
                lower_normalized_distinct=cp.distinct_count,
                normalization_reduces=False,
                sample_values=[str(v) for v in sample_values[:20]],
            )

        if string_profiles:
            results[key] = string_profiles

    return results


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

_CONFIG = {
    "thresholds": {
        "name_similarity_min": 0.6,
        "type_compatibility": "strict",
        "value_overlap_min": 3,
        "value_overlap_ratio": 0.05,
        "jaccard_min": 0.1,
        "confidence_high": 0.85,
        "confidence_medium": 0.70,
        "confidence_low": 0.50,
        "string_categorical_distinct_max": 20,
    },
    "aliases": {
        "api": ["api_number", "api_num", "well_api", "api_no", "well_number"],
        "lease": ["lease_num", "lease_number", "lease_no", "unit_num"],
        "well": ["well_name", "wellname", "well_no", "well_num"],
        "operator": ["company", "company_name", "oper", "operator_name", "opr_name"],
        "block": ["block_num", "blocknumber", "blk_num"],
        "area": ["area_code", "area_name"],
        "date": ["p_date", "prod_date", "production_date", "activity_date", "eff_date"],
        "oil": ["oil_prod", "oill", "oil_rate", "oil_volume"],
        "gas": ["gas_prod", "gass", "gas_rate", "gas_volume"],
        "water": ["water_prod", "water_rate", "water_volume"],
        "status": ["status", "well_status", "prod_status"],
    },
    "profiling": {
        "mode_a_max_rows": 100000,
        "mode_b_string_cardinality": 5000,
        "sample_size": 100,
    },
    "exclusions": {
        "column_patterns": [],
        "table_patterns": [],
    },
}


# ---------------------------------------------------------------------------
# Integration test: full pipeline on fixed snapshot
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end pipeline on a fixed database snapshot.

    Exercises all modules: metadata -> profiling -> candidates -> type compat
    -> value evidence -> string evidence -> scoring -> graph -> curation
    -> export (markdown + JSON) -> drift -> save/load.
    """

    def setup_method(self) -> None:
        self.inventory = _build_snapshot()
        self.profiles = _build_profiles(self.inventory)
        self.string_profiles = _build_string_profiles(self.inventory, self.profiles)

    # --- Phase 1: Candidates & Type Compat ---

    def test_candidate_generation_on_snapshot(self):
        candidates = generate_candidates(self.inventory, _CONFIG)
        assert len(candidates) > 0

        # Exact name matches should exist (well_id, api_number, etc.)
        exact = [c for c in candidates if c.match_type == "exact_name"]
        assert len(exact) > 0

        # Alias matches should exist (lease_num, operator, etc.)
        alias = [c for c in candidates if c.match_type == "alias"]
        assert len(alias) > 0

        # All candidates should have name_score > 0
        for c in candidates:
            assert c.name_score > 0

    # --- Phase 2: Full analysis pipeline ---

    def test_pipeline_produces_suggested_edges(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        edges = result.suggested_edges
        assert len(edges) > 0

        # All edges should have confidence and band
        for edge in edges:
            assert 0.0 <= edge["confidence"] <= 1.0
            assert edge["confidence_band"] in ("high", "medium", "low")
            assert "evidence" in edge
            assert "name_score" in edge["evidence"]
            assert "type_score" in edge["evidence"]
            assert "value_score" in edge["evidence"]

        # Edges should be sorted by confidence descending
        confidences = [e["confidence"] for e in edges]
        assert confidences == sorted(confidences, reverse=True)

    def test_pipeline_includes_value_evidence(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        # At least some edges should have value evidence
        edges_with_ve = [
            e for e in result.suggested_edges
            if "value_evidence" in e.get("evidence", {})
        ]
        assert len(edges_with_ve) > 0

        for edge in edges_with_ve:
            ve = edge["evidence"]["value_evidence"]
            assert "jaccard" in ve
            assert "exact_overlap_ratio" in ve
            assert "mode" in ve

    def test_pipeline_includes_string_evidence(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        edges_with_se = [
            e for e in result.suggested_edges
            if "string_evidence" in e.get("evidence", {})
        ]
        assert len(edges_with_se) > 0

    def test_pipeline_determinism(self):
        result1 = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )
        result2 = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        edges1 = result1.suggested_edges
        edges2 = result2.suggested_edges

        assert len(edges1) == len(edges2)
        for e1, e2 in zip(edges1, edges2):
            assert e1["source_table"] == e2["source_table"]
            assert e1["source_column"] == e2["source_column"]
            assert e1["target_table"] == e2["target_table"]
            assert e1["target_column"] == e2["target_column"]
            assert e1["confidence"] == e2["confidence"]
            assert e1["confidence_band"] == e2["confidence_band"]

    # --- Phase 3: Graph + curation ---

    def test_graph_load_from_pipeline(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        graph = RelationshipGraph()
        for key, meta in self.inventory.items():
            graph.add_table("conn1", "test_db", meta)

        # Load suggested edges
        for edge_data in result.suggested_edges:
            src_key = f"conn1.{edge_data['source_table']}"
            tgt_key = f"conn1.{edge_data['target_table']}"

            if src_key in graph.nodes and tgt_key in graph.nodes:
                edge = SuggestedEdge(
                    source_key=src_key,
                    source_column=edge_data["source_column"],
                    target_key=tgt_key,
                    target_column=edge_data["target_column"],
                    confidence=edge_data["confidence"],
                    confidence_band=edge_data["confidence_band"],
                    evidence=edge_data.get("evidence", {}),
                )
                graph.add_suggested_edge(edge)

        assert len(graph.nodes) == len(self.inventory)
        assert len(graph.suggested_edges) > 0

        # Accept top edge
        top = graph.suggested_edges[0]
        graph.confirm_edge(
            top.source_key, top.source_column,
            top.target_key, top.target_column,
            rel_type="one-to-many",
        )
        assert len(graph.confirmed_edges) == 1
        assert len(graph.suggested_edges) == len(result.suggested_edges) - 1

    # --- Phase 4: Export ---

    def test_markdown_export_full_pipeline(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        graph = RelationshipGraph()
        for key, meta in self.inventory.items():
            graph.add_table("conn1", "test_db", meta)

        for edge_data in result.suggested_edges:
            src_key = f"conn1.{edge_data['source_table']}"
            tgt_key = f"conn1.{edge_data['target_table']}"
            if src_key in graph.nodes and tgt_key in graph.nodes:
                edge = SuggestedEdge(
                    source_key=src_key,
                    source_column=edge_data["source_column"],
                    target_key=tgt_key,
                    target_column=edge_data["target_column"],
                    confidence=edge_data["confidence"],
                    confidence_band=edge_data["confidence_band"],
                    evidence=edge_data.get("evidence", {}),
                )
                graph.add_suggested_edge(edge)

        # Accept top 3 edges
        for i in range(min(3, len(graph.suggested_edges))):
            edge = graph.suggested_edges[0]
            graph.confirm_edge(
                edge.source_key, edge.source_column,
                edge.target_key, edge.target_column,
                rel_type="one-to-many",
            )

        md = generate_markdown(
            graph,
            "Wells Production Relationships",
            [{"server": "test-srv", "database": "test_db"}],
            include_evidence=True,
        )

        # Verify structure
        assert "---" in md
        assert "Wells Production Relationships" in md
        assert "table_count: 4" in md
        assert "relationship_count: 3" in md
        assert "```mermaid" in md
        assert "erDiagram" in md
        assert "## Tables" in md
        assert "## Relationships" in md
        assert "## Evidence Summary" in md
        assert "## Entity Relationship Diagram" in md

        # All tables present
        for table_name in ("wells", "production", "leases", "operators"):
            assert table_name in md

    def test_json_export_full_pipeline(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        graph = RelationshipGraph()
        for key, meta in self.inventory.items():
            graph.add_table("conn1", "test_db", meta)

        for edge_data in result.suggested_edges:
            src_key = f"conn1.{edge_data['source_table']}"
            tgt_key = f"conn1.{edge_data['target_table']}"
            if src_key in graph.nodes and tgt_key in graph.nodes:
                edge = SuggestedEdge(
                    source_key=src_key,
                    source_column=edge_data["source_column"],
                    target_key=tgt_key,
                    target_column=edge_data["target_column"],
                    confidence=edge_data["confidence"],
                    confidence_band=edge_data["confidence_band"],
                    evidence=edge_data.get("evidence", {}),
                )
                graph.add_suggested_edge(edge)

        graph.confirm_edge(
            graph.suggested_edges[0].source_key,
            graph.suggested_edges[0].source_column,
            graph.suggested_edges[0].target_key,
            graph.suggested_edges[0].target_column,
            rel_type="one-to-many",
            annotation="Primary relationship",
        )

        report = generate_json_report(
            graph,
            "Wells Production Relationships",
            [{"server": "test-srv", "database": "test_db"}],
        )

        assert report["version"] == "1.0.0"
        assert report["title"] == "Wells Production Relationships"
        assert report["summary"]["table_count"] == 4
        assert report["summary"]["confirmed_relationship_count"] == 1
        assert len(report["tables"]) == 4
        assert len(report["relationships"]) == 1
        assert report["relationships"][0]["annotation"] == "Primary relationship"
        assert len(report["suggested_relationships"]) > 0

    def test_json_export_file_write(self, tmp_path):
        graph = RelationshipGraph()
        for key, meta in self.inventory.items():
            graph.add_table("conn1", "test_db", meta)

        filepath = str(tmp_path / "relationship_report.json")
        write_json_report(graph, filepath, "Test Report")

        with open(filepath) as f:
            data = json.load(f)

        assert data["title"] == "Test Report"
        assert data["summary"]["table_count"] == 4

    # --- Phase 5: Drift detection ---

    def test_drift_no_changes(self):
        report = detect_drift(self.inventory, {
            "tables": {
                key: {
                    "schema_name": meta.schema_name,
                    "table_name": meta.table_name,
                    "row_count": meta.row_count,
                    "columns": {
                        c.name: {
                            "data_type": c.data_type,
                            "is_nullable": c.is_nullable,
                            "ordinal_position": c.ordinal_position,
                        }
                        for c in meta.columns
                    },
                }
                for key, meta in self.inventory.items()
            },
        })
        assert not report.has_drift
        assert len(report.unchanged_tables) == len(self.inventory)

    def test_drift_detects_new_table(self):
        snapshot = {
            "tables": {
                key: {
                    "schema_name": meta.schema_name,
                    "table_name": meta.table_name,
                    "row_count": meta.row_count,
                    "columns": {
                        c.name: {
                            "data_type": c.data_type,
                            "is_nullable": c.is_nullable,
                            "ordinal_position": c.ordinal_position,
                        }
                        for c in meta.columns
                    },
                }
                for key, meta in self.inventory.items()
                if key != "dbo.operators"  # Pretend operators didn't exist before
            },
        }
        report = detect_drift(self.inventory, snapshot)
        assert "dbo.operators" in report.new_tables

    def test_drift_detects_column_change(self):
        # Simulate a type change
        modified = dict(self.inventory)
        modified["dbo.wells"] = TableMetadata(
            schema_name="dbo",
            table_name="wells",
            row_count=5000,
            columns=[
                ColumnInfo(name="well_id", data_type="bigint", is_nullable=False, ordinal_position=1),
                ColumnInfo(name="api_number", data_type="nvarchar(20)", is_nullable=False, ordinal_position=2),
            ],
        )

        snapshot = {
            "tables": {
                "dbo.wells": {
                    "schema_name": "dbo",
                    "table_name": "wells",
                    "row_count": 5000,
                    "columns": {
                        "well_id": {"data_type": "int", "is_nullable": False, "ordinal_position": 1},
                        "api_number": {"data_type": "nvarchar(20)", "is_nullable": False, "ordinal_position": 2},
                    },
                },
            },
        }

        report = detect_drift(modified, snapshot)
        assert report.has_drift
        assert len(report.changed_tables) == 1
        cc = report.changed_tables[0].column_changes
        type_changes = [c for c in cc if c.change_type == "type_changed"]
        assert len(type_changes) == 1
        assert type_changes[0].old_value == "int"
        assert type_changes[0].new_value == "bigint"

    def test_drift_snapshot_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "snapshot.json")
        save_snapshot(self.inventory, filepath)

        loaded = load_snapshot(filepath)
        assert "tables" in loaded
        assert "dbo.wells" in loaded["tables"]

        # Detect drift against itself should be clean
        report = detect_drift(self.inventory, loaded)
        assert not report.has_drift

    # --- Phase 6: Save/load roundtrip ---

    def test_save_load_roundtrip(self, tmp_path):
        graph = RelationshipGraph()
        for key, meta in self.inventory.items():
            graph.add_table("conn1", "test_db", meta)

        # Add some edges and annotations
        nodes = graph.nodes
        if len(nodes) >= 2:
            graph.confirm_edge(
                nodes[0], "col_a",
                nodes[1], "col_b",
                rel_type="one-to-many",
                annotation="Test FK",
            )
        graph.set_annotation(nodes[0], "Main wells table")

        data = build_save_data(
            connections=[{"server": "test-srv", "database": "test_db"}],
            selected_tables=list(self.inventory.keys()),
            graph_dict=graph.to_dict(),
        )

        filepath = str(tmp_path / "relationships.json")
        save_relationships(filepath, data)

        loaded = load_relationships(filepath)
        assert loaded["version"] == "0.1.0"
        assert loaded["connections"][0]["server"] == "test-srv"

        graph2 = RelationshipGraph()
        graph2.from_dict(loaded["graph"])

        if len(nodes) >= 2:
            assert len(graph2.confirmed_edges) == 1
            assert graph2.confirmed_edges[0].annotation == "Test FK"
        assert graph2.get_annotation(nodes[0]) == "Main wells table"

    # --- Phase 7: End-to-end signature check ---

    def test_full_e2e_deterministic_signature(self):
        """Run the full pipeline twice and compare a deterministic signature."""

        def _run_full() -> tuple:
            result = run_analysis(
                tables=self.inventory,
                table_profiles=self.profiles,
                string_profiles=self.string_profiles,
                config=_CONFIG,
            )
            edges = result.suggested_edges
            # Build a deterministic signature from edge metadata
            sig = tuple(
                (e["source_table"], e["source_column"],
                 e["target_table"], e["target_column"],
                 e["confidence"], e["confidence_band"])
                for e in edges
            )
            return sig

        sig1 = _run_full()
        sig2 = _run_full()
        assert sig1 == sig2, "Pipeline outputs are not deterministic"
        assert len(sig1) > 0, "Pipeline produced no edges"

    # --- Phase 8: Evidence completeness on all edges ---

    def test_all_edges_have_complete_evidence(self):
        result = run_analysis(
            tables=self.inventory,
            table_profiles=self.profiles,
            string_profiles=self.string_profiles,
            config=_CONFIG,
        )

        for edge in result.suggested_edges:
            ev = edge["evidence"]
            assert "name_score" in ev, f"Missing name_score in {edge}"
            assert "type_score" in ev, f"Missing type_score in {edge}"
            assert "value_score" in ev, f"Missing value_score in {edge}"
            assert "uniqueness_score" in ev, f"Missing uniqueness_score in {edge}"
            assert "null_score" in ev, f"Missing null_score in {edge}"
            assert "match_type" in ev, f"Missing match_type in {edge}"
            assert "reason" in ev, f"Missing reason in {edge}"
