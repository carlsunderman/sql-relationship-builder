"""Tests for Phase 3: Review & Curation UI.

Covers accept/reject workflow, edge editing, manual add/remove,
annotation system, and full save/load roundtrip.
"""

import json
import os
import tempfile
from typing import Dict, Any

import pytest

from src.graph import RelationshipGraph, SuggestedEdge, ConfirmedEdge
from src.models import TableMetadata, ColumnInfo, TableProfile, ColumnProfile
from src.state import save_relationships, load_relationships, build_save_data
from src.export import generate_markdown


def _make_metadata(schema: str, table: str, columns: list, row_count: int = 100) -> TableMetadata:
    return TableMetadata(
        schema_name=schema,
        table_name=table,
        row_count=row_count,
        columns=[
            ColumnInfo(name=n, data_type=t, is_nullable=False, ordinal_position=i)
            for i, (n, t) in enumerate(columns)
        ],
    )


def _make_profile(schema: str, table: str) -> TableProfile:
    return TableProfile(
        schema_name=schema,
        table_name=table,
        row_count=100,
        column_count=2,
        columns={
            "id": ColumnProfile(
                name="id", data_type="int", row_count=100,
                non_null_count=100, null_count=0, null_ratio=0.0,
                distinct_count=100, distinct_ratio=1.0, uniqueness_ratio=1.0,
            ),
            "name": ColumnProfile(
                name="name", data_type="nvarchar", row_count=100,
                non_null_count=95, null_count=5, null_ratio=0.05,
                distinct_count=80, distinct_ratio=0.8, uniqueness_ratio=0.8,
            ),
        },
        profiling_mode="A",
    )


def _build_graph_with_tables() -> RelationshipGraph:
    """Build a graph with two tables for testing."""
    graph = RelationshipGraph()
    t1 = _make_metadata("dbo", "wells", [("id", "int"), ("api", "nvarchar"), ("name", "nvarchar")])
    t2 = _make_metadata("dbo", "production", [("id", "int"), ("well_id", "int"), ("oil", "decimal")])
    graph.add_table("conn1", "oil_db", t1)
    graph.add_table("conn1", "oil_db", t2)
    return graph


# ====================================================================
# P3-T1: Accept/Reject Workflow
# ====================================================================

class TestAcceptRejectWorkflow:
    def test_accept_moves_edge_from_suggested_to_confirmed(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="id",
            target_key=nodes[1], target_column="well_id",
            confidence=0.92, confidence_band="high",
        ))
        assert len(graph.suggested_edges) == 1
        assert len(graph.confirmed_edges) == 0

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")

        assert len(graph.suggested_edges) == 0
        assert len(graph.confirmed_edges) == 1
        assert graph.confirmed_edges[0].rel_type == "one-to-many"

    def test_dismiss_removes_suggested_edge(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="api",
            target_key=nodes[1], target_column="id",
            confidence=0.45, confidence_band="low",
        ))
        graph.dismiss_suggestion(nodes[0], "api", nodes[1], "id")

        assert len(graph.suggested_edges) == 0
        assert len(graph.confirmed_edges) == 0

    def test_multiple_suggestions_accepted_sequentially(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="id",
            target_key=nodes[1], target_column="well_id",
            confidence=0.92, confidence_band="high",
        ))
        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="api",
            target_key=nodes[1], target_column="id",
            confidence=0.75, confidence_band="medium",
        ))
        assert len(graph.suggested_edges) == 2

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        assert len(graph.suggested_edges) == 1
        assert len(graph.confirmed_edges) == 1

        graph.confirm_edge(nodes[0], "api", nodes[1], "id", rel_type="one-to-one")
        assert len(graph.suggested_edges) == 0
        assert len(graph.confirmed_edges) == 2

    def test_accept_updates_existing_confirmed_edge(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        assert graph.confirmed_edges[0].rel_type == "one-to-many"

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="many-to-one")
        assert len(graph.confirmed_edges) == 1
        assert graph.confirmed_edges[0].rel_type == "many-to-one"


# ====================================================================
# P3-T2: Edge Editing
# ====================================================================

class TestEdgeEditing:
    def test_edit_relationship_type(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-one")

        assert graph.confirmed_edges[0].rel_type == "one-to-one"

    def test_edit_annotation(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many",
                          annotation="FK: well_id references wells.id")

        assert graph.confirmed_edges[0].annotation == "FK: well_id references wells.id"

    def test_confirmed_edge_data_structure(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="test")
        edge = graph.confirmed_edges[0]

        assert edge.source_key == nodes[0]
        assert edge.source_column == "id"
        assert edge.target_key == nodes[1]
        assert edge.target_column == "well_id"
        assert edge.rel_type == "one-to-many"
        assert edge.annotation == "test"


# ====================================================================
# P3-T3: Manual Add Relationship
# ====================================================================

class TestManualAddRelationship:
    def test_add_relationship_between_existing_tables(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "api", nodes[1], "id", rel_type="one-to-one")

        assert len(graph.confirmed_edges) == 1
        edge = graph.confirmed_edges[0]
        assert edge.source_column == "api"
        assert edge.target_column == "id"
        assert edge.rel_type == "one-to-one"

    def test_add_relationship_preserves_existing(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        graph.confirm_edge(nodes[0], "name", nodes[1], "id", rel_type="many-to-one")

        assert len(graph.confirmed_edges) == 2

    def test_add_relationship_all_types(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        types = ["one-to-one", "one-to-many", "many-to-one", "many-to-many"]
        for i, rel_type in enumerate(types):
            graph.confirm_edge(
                nodes[0], "id", nodes[1], "well_id",
                rel_type=rel_type, annotation=f"type {i}"
            )
            assert graph.confirmed_edges[0].rel_type == rel_type

    def test_add_relationship_creates_graph_edge(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        assert not graph.graph.has_edge(nodes[0], nodes[1])

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        assert graph.graph.has_edge(nodes[0], nodes[1])


# ====================================================================
# P3-T4: Remove Relationship
# ====================================================================

class TestRemoveRelationship:
    def test_remove_confirmed_edge(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        assert len(graph.confirmed_edges) == 1

        graph.remove_edge(nodes[0], nodes[1])
        assert len(graph.confirmed_edges) == 0
        assert not graph.graph.has_edge(nodes[0], nodes[1])

    def test_remove_edge_preserves_other_edges(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        graph.confirm_edge(nodes[0], "api", nodes[1], "id", rel_type="one-to-one")

        graph.remove_edge(nodes[0], nodes[1])
        # remove_edge removes ALL edges between these two nodes
        assert len(graph.confirmed_edges) == 0

    def test_remove_nonexistent_edge_no_error(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.remove_edge(nodes[0], nodes[1])
        assert len(graph.confirmed_edges) == 0

    def test_remove_table_cleans_all_edges(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="api",
            target_key=nodes[1], target_column="id",
            confidence=0.8, confidence_band="high",
        ))

        graph.remove_table(nodes[0])
        assert nodes[0] not in graph.nodes
        assert len(graph.confirmed_edges) == 0
        assert len(graph.suggested_edges) == 0


# ====================================================================
# P3-T5: Annotation System
# ====================================================================

class TestAnnotationSystem:
    def test_node_annotation_set_and_get(self):
        graph = RelationshipGraph()
        graph.set_annotation("conn1.dbo.wells", "Well master table")
        assert graph.get_annotation("conn1.dbo.wells") == "Well master table"

    def test_edge_annotation_set_and_get(self):
        graph = RelationshipGraph()
        edge_key = "conn1.dbo.wells->conn1.dbo.production"
        graph.set_annotation(edge_key, "Production records linked to wells")
        assert graph.get_annotation(edge_key) == "Production records linked to wells"

    def test_annotation_empty_for_missing_key(self):
        graph = RelationshipGraph()
        assert graph.get_annotation("nonexistent") == ""

    def test_annotation_overwrite(self):
        graph = RelationshipGraph()
        key = "conn1.dbo.wells"
        graph.set_annotation(key, "First annotation")
        graph.set_annotation(key, "Updated annotation")
        assert graph.get_annotation(key) == "Updated annotation"

    def test_edge_annotation_in_confirmed_edge(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="FK constraint")
        assert graph.confirmed_edges[0].annotation == "FK constraint"

    def test_multiple_annotations_persist(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.set_annotation(nodes[0], "Wells table")
        graph.set_annotation(nodes[1], "Production table")
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="FK")
        edge_key = f"{nodes[0]}->{nodes[1]}"
        graph.set_annotation(edge_key, "Rel annotation")

        assert graph.get_annotation(nodes[0]) == "Wells table"
        assert graph.get_annotation(nodes[1]) == "Production table"
        assert graph.get_annotation(edge_key) == "Rel annotation"


# ====================================================================
# P3-T6: JSON Save/Load Roundtrip
# ====================================================================

class TestSaveLoadRoundtrip:
    def test_full_relationships_save_and_load(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        # Build a complete relationship state
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="FK: well_id -> wells.id")
        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="api",
            target_key=nodes[1], target_column="id",
            confidence=0.75, confidence_band="medium",
            evidence={"name_score": 0.5, "type_score": 0.3},
        ))
        graph.set_annotation(nodes[0], "Well master table")
        graph.set_annotation(nodes[1], "Production records")

        data = build_save_data(
            connections=[{"server": "localhost", "database": "oil_db"}],
            selected_tables=[nodes[0], nodes[1]],
            graph_dict=graph.to_dict(),
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            json.dump(data, f)

        try:
            loaded = load_relationships(path)

            # Verify top-level structure
            assert loaded["version"] == "0.1.0"
            assert len(loaded["connections"]) == 1
            assert loaded["connections"][0]["server"] == "localhost"
            assert len(loaded["selected_tables"]) == 2

            # Verify graph state
            graph_dict = loaded["graph"]
            assert len(graph_dict["confirmed_edges"]) == 1
            assert len(graph_dict["suggested_edges"]) == 1
            assert len(graph_dict["annotations"]) == 2

            # Verify confirmed edge details
            ce = graph_dict["confirmed_edges"][0]
            assert ce["rel_type"] == "one-to-many"
            assert ce["annotation"] == "FK: well_id -> wells.id"

            # Verify suggested edge details
            se = graph_dict["suggested_edges"][0]
            assert se["confidence"] == 0.75
            assert se["confidence_band"] == "medium"

            # Verify annotations
            assert "Well master table" in graph_dict["annotations"].values()
            assert "Production records" in graph_dict["annotations"].values()
        finally:
            os.unlink(path)

    def test_restore_graph_from_saved_dict(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="test")
        graph.set_annotation(nodes[0], "Annotation A")

        data = graph.to_dict()

        # Restore into fresh graph
        graph2 = RelationshipGraph()
        graph2.from_dict(data)

        assert len(graph2.confirmed_edges) == 1
        assert graph2.confirmed_edges[0].rel_type == "one-to-many"
        assert graph2.confirmed_edges[0].annotation == "test"
        assert graph2.get_annotation(nodes[0]) == "Annotation A"

    def test_save_excludes_credentials(self):
        data = build_save_data(
            connections=[{"server": "localhost", "database": "test_db"}],
            selected_tables=[],
            graph_dict={},
        )

        # Ensure no password or username fields
        for conn in data["connections"]:
            assert "password" not in conn
            assert "username" not in conn
            assert "password" not in json.dumps(conn)

    def test_load_nonexistent_file_returns_empty(self):
        result = load_relationships("/nonexistent/path/relationships.json")
        assert result == {}

    def test_save_load_preserves_evidence(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("c1", "db", t1)
        k2 = graph.add_table("c1", "db", t2)

        graph.add_suggested_edge(SuggestedEdge(
            source_key=k1, source_column="x",
            target_key=k2, target_column="x",
            confidence=0.85, confidence_band="high",
            evidence={
                "name_score": 0.9,
                "type_score": 1.0,
                "value_score": 0.8,
                "value_evidence": {
                    "exact_overlap_count": 50,
                    "exact_overlap_ratio": 0.5,
                    "jaccard": 0.5,
                },
            },
        ))

        data = graph.to_dict()
        graph2 = RelationshipGraph()
        graph2.from_dict(data)

        assert len(graph2.suggested_edges) == 1
        restored = graph2.suggested_edges[0]
        assert restored.evidence["name_score"] == 0.9
        assert restored.evidence["value_evidence"]["jaccard"] == 0.5


# ====================================================================
# P3-T7: Full Curation Workflow Integration
# ====================================================================

class TestFullCurationWorkflow:
    """Simulate the full user curation workflow:
    load tables -> run analysis -> review -> accept/reject -> edit -> add -> save -> reload
    """

    def test_end_to_end_curation(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        # Simulate analysis pipeline adding suggestions
        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="id",
            target_key=nodes[1], target_column="well_id",
            confidence=0.92, confidence_band="high",
            evidence={"name_score": 0.8, "type_score": 1.0},
        ))
        graph.add_suggested_edge(SuggestedEdge(
            source_key=nodes[0], source_column="api",
            target_key=nodes[1], target_column="id",
            confidence=0.45, confidence_band="low",
        ))
        assert len(graph.suggested_edges) == 2

        # User accepts high-confidence edge
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        assert len(graph.suggested_edges) == 1
        assert len(graph.confirmed_edges) == 1

        # User dismisses low-confidence edge
        graph.dismiss_suggestion(nodes[0], "api", nodes[1], "id")
        assert len(graph.suggested_edges) == 0

        # User edits relationship type
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="many-to-one", annotation="Production belongs to well")
        assert graph.confirmed_edges[0].rel_type == "many-to-one"

        # User adds manual relationship
        graph.confirm_edge(nodes[0], "name", nodes[1], "id", rel_type="one-to-one")
        assert len(graph.confirmed_edges) == 2

        # User annotates tables
        graph.set_annotation(nodes[0], "Master well registry")
        graph.set_annotation(nodes[1], "Daily production logs")

        # Save
        data = build_save_data(
            connections=[{"server": "srv", "database": "db"}],
            selected_tables=nodes,
            graph_dict=graph.to_dict(),
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            json.dump(data, f)

        try:
            # Reload into fresh graph
            loaded = load_relationships(path)
            graph2 = RelationshipGraph()
            graph2.from_dict(loaded["graph"])

            # Verify full state preserved
            assert len(graph2.confirmed_edges) == 2
            assert len(graph2.suggested_edges) == 0
            assert graph2.get_annotation(nodes[0]) == "Master well registry"
            assert graph2.get_annotation(nodes[1]) == "Daily production logs"

            # Verify edge details
            types = {e.rel_type for e in graph2.confirmed_edges}
            assert "many-to-one" in types
            assert "one-to-one" in types
        finally:
            os.unlink(path)

    def test_export_markdown_includes_curation(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="FK")
        graph.set_annotation(nodes[0], "Wells table")

        md = generate_markdown(graph, "Test Relationships",
                              databases=[{"server": "srv", "database": "db"}])

        assert "Test Relationships" in md
        assert "wells" in md
        assert "production" in md
        assert "one-to-many" in md
        assert "FK" in md
        assert "Wells table" in md
        assert "```mermaid" in md
        assert "erDiagram" in md

    def test_workflow_with_remove_and_restore(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-many")
        assert len(graph.confirmed_edges) == 1

        # Remove
        graph.remove_edge(nodes[0], nodes[1])
        assert len(graph.confirmed_edges) == 0

        # Re-add with different type
        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id", rel_type="one-to-one")
        assert len(graph.confirmed_edges) == 1
        assert graph.confirmed_edges[0].rel_type == "one-to-one"


class TestEdgeCases:
    def test_duplicate_suggested_edge_ignored(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        edge = SuggestedEdge(
            source_key=nodes[0], source_column="id",
            target_key=nodes[1], target_column="well_id",
            confidence=0.9, confidence_band="high",
        )
        graph.add_suggested_edge(edge)
        graph.add_suggested_edge(edge)

        assert len(graph.suggested_edges) == 1

    def test_confirm_edge_with_empty_annotation(self):
        graph = _build_graph_with_tables()
        nodes = graph.nodes

        graph.confirm_edge(nodes[0], "id", nodes[1], "well_id",
                          rel_type="one-to-many", annotation="")
        assert graph.confirmed_edges[0].annotation == ""

    def test_graph_to_dict_empty_graph(self):
        graph = RelationshipGraph()
        data = graph.to_dict()
        assert data == {
            "confirmed_edges": [],
            "suggested_edges": [],
            "annotations": {},
        }

    def test_from_dict_with_missing_fields(self):
        graph = RelationshipGraph()
        graph.from_dict({})
        assert len(graph.confirmed_edges) == 0
        assert len(graph.suggested_edges) == 0
        assert graph.annotations == {}
