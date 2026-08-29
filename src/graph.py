"""NetworkX graph management for SQL table relationships.

Manages nodes (tables) and edges (relationships) with support for
confirmed (FK/user-accepted) and suggested (analysis-pipeline) edges.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.models import TableMetadata


@dataclass
class SuggestedEdge:
    """A candidate relationship suggested by the analysis pipeline."""
    source_key: str
    source_column: str
    target_key: str
    target_column: str
    confidence: float = 0.0
    confidence_band: str = "low"
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.source_key, self.target_key)


@dataclass
class ConfirmedEdge:
    """A confirmed relationship accepted by the user or from FK metadata."""
    source_key: str
    source_column: str
    target_key: str
    target_column: str
    rel_type: str = "one-to-many"
    annotation: str = ""

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.source_key, self.target_key)


class RelationshipGraph:
    """Manages the relationship graph with nodes, confirmed edges, and suggested edges."""

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.suggested_edges: List[SuggestedEdge] = []
        self.confirmed_edges: List[ConfirmedEdge] = []
        self.annotations: Dict[str, str] = {}  # node_key or "source->target" -> text

    # ---- Node management ----

    def add_table(self, connection_id: str, database: str, metadata: TableMetadata) -> str:
        """Add a table as a node in the graph.

        Args:
            connection_id: Identifier for the source connection.
            database: Database name.
            metadata: Full table metadata.

        Returns:
            The node key.
        """
        node_key = f"{connection_id}.{metadata.full_name}"
        self.graph.add_node(
            node_key,
            connection_id=connection_id,
            database=database,
            schema=metadata.schema_name,
            table=metadata.table_name,
            row_count=metadata.row_count,
            columns={c.name: c.data_type for c in metadata.columns},
            primary_keys=metadata.primary_key_columns,
        )
        return node_key

    def get_node(self, node_key: str) -> Optional[Dict[str, Any]]:
        """Get node attributes by key.

        Args:
            node_key: Fully qualified node key.

        Returns:
            Node attributes dict, or None if not found.
        """
        if node_key not in self.graph:
            return None
        return dict(self.graph.nodes[node_key])

    @property
    def nodes(self) -> List[str]:
        return list(self.graph.nodes())

    def remove_table(self, node_key: str) -> None:
        """Remove a table and all its edges from the graph.

        Args:
            node_key: Node key to remove.
        """
        if node_key in self.graph:
            self.graph.remove_node(node_key)
        self.confirmed_edges = [e for e in self.confirmed_edges
                                if e.source_key != node_key and e.target_key != node_key]
        self.suggested_edges = [e for e in self.suggested_edges
                                if e.source_key != node_key and e.target_key != node_key]

    # ---- Suggested edge management ----

    def add_suggested_edge(self, edge: SuggestedEdge) -> None:
        """Add a candidate relationship from the analysis pipeline.

        Args:
            edge: The suggested edge with evidence and confidence.
        """
        for existing in self.suggested_edges:
            if (existing.source_key == edge.source_key
                    and existing.target_key == edge.target_key
                    and existing.source_column == edge.source_column
                    and existing.target_column == edge.target_column):
                return
        self.suggested_edges.append(edge)

    def add_suggested_edges(self, edges: List[SuggestedEdge]) -> None:
        """Add multiple suggested edges.

        Args:
            edges: List of suggested edges.
        """
        for edge in edges:
            self.add_suggested_edge(edge)

    # ---- Confirmed edge management ----

    def confirm_edge(
        self,
        source_key: str,
        source_column: str,
        target_key: str,
        target_column: str,
        rel_type: str = "one-to-many",
        annotation: str = "",
    ) -> None:
        """Move a suggested edge to confirmed, or add a new confirmed edge.

        Args:
            source_key: Source table node key.
            source_column: Source column name.
            target_key: Target table node key.
            target_column: Target column name.
            rel_type: Relationship type (one-to-one, one-to-many, many-to-one, many-to-many).
            annotation: Optional description.
        """
        # Remove from suggested if present
        self.suggested_edges = [
            e for e in self.suggested_edges
            if not (e.source_key == source_key
                    and e.target_key == target_key
                    and e.source_column == source_column
                    and e.target_column == target_column)
        ]

        # Check if already confirmed
        for existing in self.confirmed_edges:
            if (existing.source_key == source_key
                    and existing.target_key == target_key
                    and existing.source_column == source_column
                    and existing.target_column == target_column):
                existing.rel_type = rel_type
                existing.annotation = annotation
                return

        edge = ConfirmedEdge(
            source_key=source_key,
            source_column=source_column,
            target_key=target_key,
            target_column=target_column,
            rel_type=rel_type,
            annotation=annotation,
        )
        self.confirmed_edges.append(edge)
        self.graph.add_edge(source_key, target_key, type=rel_type)

    def remove_edge(self, source_key: str, target_key: str) -> None:
        """Remove a confirmed edge from the graph.

        Args:
            source_key: Source table node key.
            target_key: Target table node key.
        """
        self.confirmed_edges = [
            e for e in self.confirmed_edges
            if not (e.source_key == source_key and e.target_key == target_key)
        ]
        if self.graph.has_edge(source_key, target_key):
            self.graph.remove_edge(source_key, target_key)

    def dismiss_suggestion(
        self,
        source_key: str,
        source_column: str,
        target_key: str,
        target_column: str,
    ) -> None:
        """Remove a single suggested edge without confirming it.

        Args:
            source_key: Source table node key.
            source_column: Source column name.
            target_key: Target table node key.
            target_column: Target column name.
        """
        self.suggested_edges = [
            e for e in self.suggested_edges
            if not (
                e.source_key == source_key
                and e.source_column == source_column
                and e.target_key == target_key
                and e.target_column == target_column
            )
        ]

    # ---- Annotations ----

    def set_annotation(self, key: str, text: str) -> None:
        """Set a free-text annotation on a node or edge.

        Args:
            key: Node key or "source->target" edge key.
            text: Annotation text.
        """
        self.annotations[key] = text

    def get_annotation(self, key: str) -> str:
        """Get annotation for a node or edge.

        Args:
            key: Node key or "source->target" edge key.

        Returns:
            Annotation text, or empty string.
        """
        return self.annotations.get(key, "")

    # ---- Pyvis visualization ----

    def to_pyvis(
        self,
        max_suggested: int = 100,
        min_confidence: float = 0.0,
    ) -> Any:
        """Build a pyvis Network visualization from the graph.

        Only the top ``max_suggested`` suggested edges (sorted by confidence
        descending) that meet ``min_confidence`` are rendered.  All confirmed
        edges are always included.  This prevents the browser from receiving
        thousands of dashed edges when the pipeline produces a large candidate
        set.

        Args:
            max_suggested: Maximum number of suggested edges to render.
                Set to 0 to hide all suggested edges.
            min_confidence: Minimum confidence score; suggested edges below
                this value are excluded before the top-N cap is applied.

        Returns:
            Tuple of (pyvis Network object, total_suggested, visible_suggested).
        """
        from pyvis.network import Network

        net = Network(height="500px", width="100%", directed=True)
        net.barnes_hut()

        # Color palette for connections
        colors = {
            "conn1": "#4A90E2",
            "conn2": "#E74C3C",
            "conn3": "#2ECC71",
        }

        # Add nodes
        for node_key, attrs in self.graph.nodes(data=True):
            conn_id = attrs.get("connection_id", "")
            color = colors.get(conn_id, "#95A5A6")

            label = f"{attrs.get('schema', '')}.{attrs.get('table', '')}"
            title = (
                f"Database: {attrs.get('database', '')}<br>"
                f"Rows: {attrs.get('row_count', 0):,}<br>"
                f"Columns: {len(attrs.get('columns', {}))}"
            )
            net.add_node(node_key, label=label, color=color, title=title)

        # Add confirmed edges (solid) — always all of them
        for edge in self.confirmed_edges:
            edge_key = f"{edge.source_key}->{edge.target_key}"
            annotation = self.get_annotation(edge_key) or edge.annotation
            title_parts = [
                f"Type: {edge.rel_type}",
                f"{edge.source_column} -> {edge.target_column}",
            ]
            if annotation:
                title_parts.append(f"Note: {annotation}")
            net.add_edge(
                edge.source_key, edge.target_key,
                label=edge.rel_type,
                width=2,
                title="<br>".join(title_parts),
            )

        # Apply confidence floor then cap at max_suggested (top-N by confidence).
        total_suggested = len(self.suggested_edges)
        eligible = [
            e for e in self.suggested_edges if e.confidence >= min_confidence
        ]
        eligible.sort(key=lambda e: e.confidence, reverse=True)
        visible = eligible[:max_suggested] if max_suggested > 0 else []

        # Add suggested edges (dashed)
        band_colors = {
            "high":   "#E67E22",
            "medium": "#F1C40F",
            "low":    "#BDC3C7",
        }
        for edge in visible:
            color = band_colors.get(edge.confidence_band, "#F1C40F")
            net.add_edge(
                edge.source_key, edge.target_key,
                label=f"{edge.confidence_band} ({edge.confidence:.2f})",
                width=1,
                dashes=[5, 5],
                color=color,
                title=(
                    f"Confidence: {edge.confidence:.2f}<br>"
                    f"Band: {edge.confidence_band}"
                ),
            )

        return net, total_suggested, len(visible)

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph state to a dictionary for JSON persistence.

        Returns:
            Serializable dictionary.
        """
        return {
            "confirmed_edges": [
                {
                    "source_key": e.source_key,
                    "source_column": e.source_column,
                    "target_key": e.target_key,
                    "target_column": e.target_column,
                    "rel_type": e.rel_type,
                    "annotation": e.annotation,
                }
                for e in self.confirmed_edges
            ],
            "suggested_edges": [
                {
                    "source_key": e.source_key,
                    "source_column": e.source_column,
                    "target_key": e.target_key,
                    "target_column": e.target_column,
                    "confidence": e.confidence,
                    "confidence_band": e.confidence_band,
                    "evidence": e.evidence,
                }
                for e in self.suggested_edges
            ],
            "annotations": dict(self.annotations),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore graph state from a dictionary.

        Args:
            data: Dictionary produced by to_dict().
        """
        for e_data in data.get("confirmed_edges", []):
            self.confirmed_edges.append(ConfirmedEdge(
                source_key=e_data["source_key"],
                source_column=e_data.get("source_column", ""),
                target_key=e_data["target_key"],
                target_column=e_data.get("target_column", ""),
                rel_type=e_data.get("rel_type", "one-to-many"),
                annotation=e_data.get("annotation", ""),
            ))
        for e_data in data.get("suggested_edges", []):
            self.suggested_edges.append(SuggestedEdge(
                source_key=e_data["source_key"],
                source_column=e_data.get("source_column", ""),
                target_key=e_data["target_key"],
                target_column=e_data.get("target_column", ""),
                confidence=e_data.get("confidence", 0.0),
                confidence_band=e_data.get("confidence_band", "low"),
                evidence=e_data.get("evidence", {}),
            ))
        self.annotations.update(data.get("annotations", {}))