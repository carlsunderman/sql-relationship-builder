"""Tests for relationship graph management."""
from src.graph import (
    RelationshipGraph,
    SuggestedEdge,
    ConfirmedEdge,
)
from src.models import TableMetadata, ColumnInfo


def _make_metadata(schema: str, table: str, columns: list) -> TableMetadata:
    return TableMetadata(
        schema_name=schema,
        table_name=table,
        row_count=100,
        columns=[ColumnInfo(name=n, data_type=t, is_nullable=False, ordinal_position=i)
                 for i, (n, t) in enumerate(columns)],
    )


class TestRelationshipGraph:
    def test_add_table_creates_node(self):
        graph = RelationshipGraph()
        meta = _make_metadata("dbo", "customers", [("id", "int")])
        key = graph.add_table("conn1", "sales", meta)
        assert key in graph.nodes
        node = graph.get_node(key)
        assert node is not None
        assert node["table"] == "customers"

    def test_add_suggested_edge(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("c1", "db", t1)
        k2 = graph.add_table("c1", "db", t2)

        edge = SuggestedEdge(
            source_key=k1, source_column="x",
            target_key=k2, target_column="x",
            confidence=0.85, confidence_band="high",
        )
        graph.add_suggested_edge(edge)
        assert len(graph.suggested_edges) == 1

    def test_confirm_edge_moves_from_suggested(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("c1", "db", t1)
        k2 = graph.add_table("c1", "db", t2)

        edge = SuggestedEdge(source_key=k1, source_column="x",
                             target_key=k2, target_column="x")
        graph.add_suggested_edge(edge)
        graph.confirm_edge(k1, "x", k2, "x", rel_type="one-to-many")

        assert len(graph.suggested_edges) == 0
        assert len(graph.confirmed_edges) == 1
        assert graph.graph.has_edge(k1, k2)

    def test_remove_edge(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("c1", "db", t1)
        k2 = graph.add_table("c1", "db", t2)

        graph.confirm_edge(k1, "x", k2, "x", rel_type="one-to-many")
        graph.remove_edge(k1, k2)

        assert len(graph.confirmed_edges) == 0
        assert not graph.graph.has_edge(k1, k2)

    def test_annotations(self):
        graph = RelationshipGraph()
        graph.set_annotation("my_node", "A test annotation")
        assert graph.get_annotation("my_node") == "A test annotation"
        assert graph.get_annotation("nonexistent") == ""

    def test_remove_table_cleans_edges(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("c1", "db", t1)
        k2 = graph.add_table("c1", "db", t2)

        graph.confirm_edge(k1, "x", k2, "x", rel_type="one-to-many")
        graph.add_suggested_edge(SuggestedEdge(
            source_key=k1, source_column="y",
            target_key=k2, target_column="y",
        ))
        graph.remove_table(k1)

        assert k1 not in graph.nodes
        assert len(graph.confirmed_edges) == 0
        assert len(graph.suggested_edges) == 0


class TestSerialization:
    def test_to_dict_roundtrip(self):
        graph = RelationshipGraph()
        t1 = _make_metadata("dbo", "a", [("x", "int")])
        t2 = _make_metadata("dbo", "b", [("x", "int")])
        k1 = graph.add_table("c1", "db", t1)
        k2 = graph.add_table("c1", "db", t2)

        graph.confirm_edge(k1, "x", k2, "x", rel_type="one-to-many")
        graph.set_annotation(k1, "Table A description")

        data = graph.to_dict()
        assert len(data["confirmed_edges"]) == 1
        assert "Table A description" in data["annotations"].values()

        # Restore into a fresh graph
        graph2 = RelationshipGraph()
        graph2.from_dict(data)
        assert len(graph2.confirmed_edges) == 1
        assert graph2.get_annotation(k1) == "Table A description"