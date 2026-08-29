"""Tests for markdown and JSON export."""

import json
import os
import tempfile

from src.export import (
    generate_json_report,
    generate_markdown,
    write_json_report,
    _rel_type_to_mermaid,
    _short_label,
)
from src.graph import RelationshipGraph, SuggestedEdge
from src.models import ColumnInfo, TableMetadata


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
# Markdown export
# ---------------------------------------------------------------------------

class TestGenerateMarkdown:
    def test_basic_structure(self):
        graph = RelationshipGraph()
        meta = _make_metadata("dbo", "customers", [("id", "int"), ("name", "nvarchar")])
        graph.add_table("conn1", "sales", meta)

        md = generate_markdown(graph, "Test Relationships", [{"server": "srv1", "database": "sales"}])
        assert "---" in md
        assert "Test Relationships" in md
        assert "customers" in md
        assert "```mermaid" in md
        assert "erDiagram" in md
        assert "table_count:" in md
        assert "relationship_count:" in md
        assert "generated:" in md

    def test_relationships_included(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "customers", [("id", "int")])
        t2 = _make_metadata("dbo", "orders", [("id", "int"), ("customer_id", "int")])
        k1 = graph.add_table("conn1", "db", t1)
        k2 = graph.add_table("conn1", "db", t2)

        graph.confirm_edge(k1, "id", k2, "customer_id", rel_type="one-to-many")

        md = generate_markdown(graph, "Test")
        assert "one-to-many" in md
        assert "customers" in md
        assert "orders" in md

    def test_suggested_edges_section(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("conn1", "db", t1)
        k2 = graph.add_table("conn1", "db", t2)

        graph.add_suggested_edge(SuggestedEdge(
            source_key=k1, source_column="x",
            target_key=k2, target_column="x",
            confidence=0.85, confidence_band="high",
        ))

        md = generate_markdown(graph)
        assert "Suggested Relationships (Pending Review)" in md
        assert "0.85" in md

    def test_evidence_summary(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("conn1", "db", t1)
        k2 = graph.add_table("conn1", "db", t2)

        graph.add_suggested_edge(SuggestedEdge(
            source_key=k1, source_column="x",
            target_key=k2, target_column="x",
            confidence=0.85,
            confidence_band="high",
            evidence={
                "name_score": 1.0,
                "type_score": 1.0,
                "value_score": 0.8,
                "uniqueness_score": 0.9,
                "null_score": 0.95,
                "match_type": "exact_name",
                "reason": "Exact name match: x",
            },
        ))

        md = generate_markdown(graph, include_evidence=True)
        assert "Evidence Summary" in md
        assert "Name Score:" in md
        assert "Type Score:" in md
        assert "Value Score:" in md

    def test_evidence_summary_disabled(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("conn1", "db", t1)
        k2 = graph.add_table("conn1", "db", t2)

        graph.add_suggested_edge(SuggestedEdge(
            source_key=k1, source_column="x",
            target_key=k2, target_column="x",
            confidence=0.85, confidence_band="high",
            evidence={"name_score": 1.0},
        ))

        md = generate_markdown(graph, include_evidence=False)
        assert "Evidence Summary" not in md

    def test_annotations_section(self):
        graph = RelationshipGraph()
        meta = _make_metadata("dbo", "t1", [("id", "int")])
        k1 = graph.add_table("conn1", "db", meta)
        graph.set_annotation(k1, "Main table")

        md = generate_markdown(graph)
        assert "## Annotations" in md
        assert "Main table" in md

    def test_empty_graph(self):
        graph = RelationshipGraph()
        md = generate_markdown(graph, "Empty")
        assert md is not None
        assert "Empty" in md


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

class TestGenerateJsonReport:
    def test_basic_structure(self):
        graph = RelationshipGraph()
        meta = _make_metadata("dbo", "customers", [("id", "int"), ("name", "nvarchar")])
        graph.add_table("conn1", "sales", meta)

        report = generate_json_report(
            graph, "Test Relationships",
            [{"server": "srv1", "database": "sales"}],
        )
        assert report["version"] == "1.0.0"
        assert report["title"] == "Test Relationships"
        assert "generated_at" in report
        assert report["summary"]["table_count"] == 1
        assert report["summary"]["confirmed_relationship_count"] == 0
        assert len(report["tables"]) == 1
        assert report["tables"][0]["table"] == "customers"

    def test_relationships_included(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "customers", [("id", "int")])
        t2 = _make_metadata("dbo", "orders", [("id", "int"), ("customer_id", "int")])
        k1 = graph.add_table("conn1", "db", t1)
        k2 = graph.add_table("conn1", "db", t2)

        graph.confirm_edge(k1, "id", k2, "customer_id", rel_type="one-to-many", annotation="FK")

        report = generate_json_report(graph)
        assert len(report["relationships"]) == 1
        rel = report["relationships"][0]
        assert rel["source_column"] == "id"
        assert rel["target_column"] == "customer_id"
        assert rel["relationship_type"] == "one-to-many"
        assert rel["annotation"] == "FK"

    def test_suggested_with_evidence(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("conn1", "db", t1)
        k2 = graph.add_table("conn1", "db", t2)

        graph.add_suggested_edge(SuggestedEdge(
            source_key=k1, source_column="x",
            target_key=k2, target_column="x",
            confidence=0.85, confidence_band="high",
            evidence={"name_score": 1.0, "type_score": 1.0},
        ))

        report = generate_json_report(graph)
        assert len(report["suggested_relationships"]) == 1
        suggested = report["suggested_relationships"][0]
        assert suggested["confidence"] == 0.85
        assert suggested["confidence_band"] == "high"
        assert suggested["evidence"]["name_score"] == 1.0

    def test_annotations_included(self):
        graph = RelationshipGraph()
        meta = _make_metadata("dbo", "t1", [("id", "int")])
        k1 = graph.add_table("conn1", "db", meta)
        graph.set_annotation(k1, "Test annotation")

        report = generate_json_report(graph)
        assert report["annotations"][k1] == "Test annotation"
        assert report["tables"][0]["annotation"] == "Test annotation"

    def test_empty_graph(self):
        graph = RelationshipGraph()
        report = generate_json_report(graph)
        assert report["summary"]["table_count"] == 0
        assert report["tables"] == []
        assert report["relationships"] == []


class TestWriteJsonReport:
    def test_write_and_read(self, tmp_path):
        graph = RelationshipGraph()
        meta = _make_metadata("dbo", "t1", [("id", "int")])
        graph.add_table("conn1", "db", meta)

        filepath = str(tmp_path / "report.json")
        write_json_report(graph, filepath, "Test")

        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["title"] == "Test"
        assert len(data["tables"]) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestRelTypeToMermaid:
    def test_one_to_one(self):
        assert _rel_type_to_mermaid("one-to-one") == ("|o", "o|")

    def test_one_to_many(self):
        assert _rel_type_to_mermaid("one-to-many") == ("|o", "o{")

    def test_many_to_one(self):
        assert _rel_type_to_mermaid("many-to-one") == ("}o", "o|")

    def test_many_to_many(self):
        assert _rel_type_to_mermaid("many-to-many") == ("}o", "o{")

    def test_fallback(self):
        assert _rel_type_to_mermaid("unknown") == ("|o", "o{")


class TestShortLabel:
    def test_three_parts(self):
        assert _short_label("conn1.dbo.customers") == "dbo.customers"

    def test_two_parts(self):
        assert _short_label("dbo.customers") == "dbo.customers"

    def test_single_part(self):
        assert _short_label("customers") == "customers"
