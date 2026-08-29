"""SQLite-backed graph store for persisted relationship graphs.

Provides CRUD for graphs, nodes, edges, and snapshots.
Implements cross-graph confirmation tracking and promotion logic.
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GRAPH_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "graph_store.db",
)

# Number of unique session confirmations required before an edge is promoted.
PROMOTION_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StoredGraph:
    """A named relationship graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    domain_tag: str = ""
    database_server: str = ""  # server\database identifier
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    is_template: bool = False
    parent_graph_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class StoredNode:
    """A table node within a graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    graph_id: str = ""
    connection_id: str = ""
    schema_name: str = ""
    table_name: str = ""
    row_count: int = 0
    columns_json: str = "{}"  # {col_name: data_type}
    descriptions_json: str = "{}"  # {col_name: description}

    @property
    def full_table_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def columns(self) -> Dict[str, str]:
        return json.loads(self.columns_json or "{}")

    @property
    def descriptions(self) -> Dict[str, str]:
        return json.loads(self.descriptions_json or "{}")


@dataclass
class StoredEdge:
    """A relationship edge within a graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    graph_id: str = ""
    source_table_key: str = ""  # schema.table
    source_column: str = ""
    target_table_key: str = ""  # schema.table
    target_column: str = ""
    edge_type: str = "suggested"  # "confirmed" | "suggested"
    confidence: float = 0.0
    rel_type: str = "one-to-many"
    annotation: str = ""
    evidence_json: str = "{}"
    origin: str = "user"  # "fk", "ai", "user", "pipeline"
    confirmed_by_count: int = 0
    confirmers_json: str = "[]"  # list of session identifiers

    @property
    def evidence(self) -> Dict[str, Any]:
        return json.loads(self.evidence_json or "{}")

    @property
    def confirmers(self) -> List[str]:
        return json.loads(self.confirmers_json or "[]")


@dataclass
class GraphSnapshot:
    """Full serialized state of a graph at a version."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    graph_id: str = ""
    version: int = 1
    snapshot_json: str = "{}"
    created_at: str = ""

    @property
    def data(self) -> Dict[str, Any]:
        return json.loads(self.snapshot_json or "{}")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS graphs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    domain_tag TEXT NOT NULL DEFAULT '',
    database_server TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    is_template INTEGER NOT NULL DEFAULT 0,
    parent_graph_id TEXT
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    connection_id TEXT NOT NULL DEFAULT '',
    schema_name TEXT NOT NULL DEFAULT '',
    table_name TEXT NOT NULL DEFAULT '',
    row_count INTEGER NOT NULL DEFAULT 0,
    columns_json TEXT NOT NULL DEFAULT '{}',
    descriptions_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_graph_id ON graph_nodes(graph_id);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    source_table_key TEXT NOT NULL DEFAULT '',
    source_column TEXT NOT NULL DEFAULT '',
    target_table_key TEXT NOT NULL DEFAULT '',
    target_column TEXT NOT NULL DEFAULT '',
    edge_type TEXT NOT NULL DEFAULT 'suggested',
    confidence REAL NOT NULL DEFAULT 0.0,
    rel_type TEXT NOT NULL DEFAULT 'one-to-many',
    annotation TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    origin TEXT NOT NULL DEFAULT 'user',
    confirmed_by_count INTEGER NOT NULL DEFAULT 0,
    confirmers_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_graph_id ON graph_edges(graph_id);

CREATE INDEX IF NOT EXISTS idx_graph_edges_signature ON graph_edges(
    source_table_key, source_column, target_table_key, target_column
);

CREATE TABLE IF NOT EXISTS graph_snapshots (
    id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_graph_snapshots_graph_id ON graph_snapshots(graph_id);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(GRAPH_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.close()


# ---------------------------------------------------------------------------
# Graph CRUD
# ---------------------------------------------------------------------------

GRAPH_COLUMNS = [
    "id", "name", "description", "domain_tag", "database_server",
    "created_by", "created_at", "updated_at", "version",
    "is_template", "parent_graph_id",
]


def _row_to_graph(row: sqlite3.Row) -> StoredGraph:
    return StoredGraph(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        domain_tag=row["domain_tag"],
        database_server=row["database_server"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        version=row["version"],
        is_template=bool(row["is_template"]),
        parent_graph_id=row["parent_graph_id"],
    )


def list_graphs(domain_tag: Optional[str] = None) -> List[StoredGraph]:
    """List all graphs, optionally filtered by domain tag."""
    conn = _get_conn()
    if domain_tag:
        rows = conn.execute(
            "SELECT * FROM graphs WHERE domain_tag = ? ORDER BY is_template DESC, updated_at DESC",
            (domain_tag,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM graphs ORDER BY is_template DESC, updated_at DESC"
        ).fetchall()
    conn.close()
    return [_row_to_graph(r) for r in rows]


def get_graph(graph_id: str) -> Optional[StoredGraph]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM graphs WHERE id = ?", (graph_id,)
    ).fetchone()
    conn.close()
    return _row_to_graph(row) if row else None


def save_graph(graph: StoredGraph) -> StoredGraph:
    """Insert or update a graph."""
    graph.updated_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        """INSERT INTO graphs
           (id, name, description, domain_tag, database_server,
            created_by, created_at, updated_at, version, is_template, parent_graph_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             name=excluded.name,
             description=excluded.description,
             domain_tag=excluded.domain_tag,
             database_server=excluded.database_server,
             updated_at=excluded.updated_at,
             version=excluded.version,
             is_template=excluded.is_template,
             parent_graph_id=excluded.parent_graph_id
        """,
        (
            graph.id, graph.name, graph.description, graph.domain_tag,
            graph.database_server, graph.created_by, graph.created_at,
            graph.updated_at, graph.version, int(graph.is_template),
            graph.parent_graph_id,
        ),
    )
    conn.commit()
    conn.close()
    return graph


def delete_graph(graph_id: str) -> None:
    """Delete a graph and all its nodes, edges, and snapshots."""
    conn = _get_conn()
    conn.execute("DELETE FROM graphs WHERE id = ?", (graph_id,))
    conn.commit()
    conn.close()


def fork_graph(source_id: str, name: str, created_by: str) -> StoredGraph:
    """Create a personal fork of an existing graph.

    Copies the graph metadata and all nodes/edges. The fork starts at version 1.
    """
    source = get_graph(source_id)
    if source is None:
        raise ValueError(f"Source graph {source_id} not found")

    fork = StoredGraph(
        name=name,
        description=source.description,
        domain_tag=source.domain_tag,
        database_server=source.database_server,
        created_by=created_by,
        parent_graph_id=source_id,
        is_template=False,
    )
    save_graph(fork)

    # Copy nodes
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM graph_nodes WHERE graph_id = ?", (source_id,)
    ).fetchall()
    for row in rows:
        node = StoredNode(
            graph_id=fork.id,
            connection_id=row["connection_id"],
            schema_name=row["schema_name"],
            table_name=row["table_name"],
            row_count=row["row_count"],
            columns_json=row["columns_json"],
            descriptions_json=row["descriptions_json"],
        )
        save_node(node)

    # Copy edges
    edge_rows = conn.execute(
        "SELECT * FROM graph_edges WHERE graph_id = ?", (source_id,)
    ).fetchall()
    for row in edge_rows:
        edge = StoredEdge(
            graph_id=fork.id,
            source_table_key=row["source_table_key"],
            source_column=row["source_column"],
            target_table_key=row["target_table_key"],
            target_column=row["target_column"],
            edge_type=row["edge_type"],
            confidence=row["confidence"],
            rel_type=row["rel_type"],
            annotation=row["annotation"],
            evidence_json=row["evidence_json"],
            origin=row["origin"],
            confirmed_by_count=row["confirmed_by_count"],
            confirmers_json=row["confirmers_json"],
        )
        save_edge(edge)

    conn.close()
    return fork


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

def list_nodes(graph_id: str) -> List[StoredNode]:
    """List all nodes in a graph."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM graph_nodes WHERE graph_id = ?", (graph_id,)
    ).fetchall()
    conn.close()
    return [
        StoredNode(
            id=r["id"], graph_id=r["graph_id"],
            connection_id=r["connection_id"],
            schema_name=r["schema_name"], table_name=r["table_name"],
            row_count=r["row_count"],
            columns_json=r["columns_json"],
            descriptions_json=r["descriptions_json"],
        )
        for r in rows
    ]


def save_node(node: StoredNode) -> StoredNode:
    """Insert or update a node."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO graph_nodes
           (id, graph_id, connection_id, schema_name, table_name,
            row_count, columns_json, descriptions_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             graph_id=excluded.graph_id,
             connection_id=excluded.connection_id,
             schema_name=excluded.schema_name,
             table_name=excluded.table_name,
             row_count=excluded.row_count,
             columns_json=excluded.columns_json,
             descriptions_json=excluded.descriptions_json
        """,
        (
            node.id, node.graph_id, node.connection_id,
            node.schema_name, node.table_name, node.row_count,
            node.columns_json, node.descriptions_json,
        ),
    )
    conn.commit()
    conn.close()
    return node


def delete_nodes_for_graph(graph_id: str) -> None:
    """Remove all nodes for a graph (used before re-saving)."""
    conn = _get_conn()
    conn.execute("DELETE FROM graph_nodes WHERE graph_id = ?", (graph_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

def list_edges(graph_id: str) -> List[StoredEdge]:
    """List all edges in a graph."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM graph_edges WHERE graph_id = ?", (graph_id,)
    ).fetchall()
    conn.close()
    return [
        StoredEdge(
            id=r["id"], graph_id=r["graph_id"],
            source_table_key=r["source_table_key"],
            source_column=r["source_column"],
            target_table_key=r["target_table_key"],
            target_column=r["target_column"],
            edge_type=r["edge_type"], confidence=r["confidence"],
            rel_type=r["rel_type"], annotation=r["annotation"],
            evidence_json=r["evidence_json"], origin=r["origin"],
            confirmed_by_count=r["confirmed_by_count"],
            confirmers_json=r["confirmers_json"],
        )
        for r in rows
    ]


def save_edge(edge: StoredEdge) -> StoredEdge:
    """Insert or update an edge."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO graph_edges
           (id, graph_id, source_table_key, source_column,
            target_table_key, target_column, edge_type, confidence,
            rel_type, annotation, evidence_json, origin,
            confirmed_by_count, confirmers_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             graph_id=excluded.graph_id,
             source_table_key=excluded.source_table_key,
             source_column=excluded.source_column,
             target_table_key=excluded.target_table_key,
             target_column=excluded.target_column,
             edge_type=excluded.edge_type,
             confidence=excluded.confidence,
             rel_type=excluded.rel_type,
             annotation=excluded.annotation,
             evidence_json=excluded.evidence_json,
             origin=excluded.origin,
             confirmed_by_count=excluded.confirmed_by_count,
             confirmers_json=excluded.confirmers_json
        """,
        (
            edge.id, edge.graph_id, edge.source_table_key, edge.source_column,
            edge.target_table_key, edge.target_column, edge.edge_type,
            edge.confidence, edge.rel_type, edge.annotation,
            edge.evidence_json, edge.origin,
            edge.confirmed_by_count, edge.confirmers_json,
        ),
    )
    conn.commit()
    conn.close()
    return edge


def delete_edges_for_graph(graph_id: str) -> None:
    """Remove all edges for a graph (used before re-saving)."""
    conn = _get_conn()
    conn.execute("DELETE FROM graph_edges WHERE graph_id = ?", (graph_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Snapshot CRUD
# ---------------------------------------------------------------------------

def save_snapshot(snapshot: GraphSnapshot) -> GraphSnapshot:
    """Save a full graph snapshot."""
    snapshot.created_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO graph_snapshots (id, graph_id, version, snapshot_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            snapshot.id, snapshot.graph_id, snapshot.version,
            snapshot.snapshot_json, snapshot.created_at,
        ),
    )
    conn.commit()
    conn.close()
    return snapshot


def get_latest_snapshot(graph_id: str) -> Optional[GraphSnapshot]:
    """Get the latest snapshot for a graph."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM graph_snapshots WHERE graph_id = ? "
        "ORDER BY version DESC LIMIT 1",
        (graph_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return GraphSnapshot(
        id=row["id"], graph_id=row["graph_id"],
        version=row["version"], snapshot_json=row["snapshot_json"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Confirmation tracking and promotion
# ---------------------------------------------------------------------------

def record_edge_confirmation(
    graph_id: str,
    source_table_key: str,
    source_column: str,
    target_table_key: str,
    target_column: str,
    session_id: str,
) -> int:
    """Record that a session confirmed an edge.

    Updates the confirmers list across ALL graphs that have this edge signature.
    Returns the total confirmation count.
    """
    conn = _get_conn()
    cursor = conn.cursor()

    # Find all edges matching this signature across all graphs
    rows = cursor.execute(
        "SELECT id, confirmers_json FROM graph_edges "
        "WHERE source_table_key = ? AND source_column = ? "
        "AND target_table_key = ? AND target_column = ?",
        (source_table_key, source_column, target_table_key, target_column),
    ).fetchall()

    for row in rows:
        confirmers: List[str] = json.loads(row["confirmers_json"] or "[]")
        if session_id not in confirmers:
            confirmers.append(session_id)
            cursor.execute(
                "UPDATE graph_edges SET confirmers_json = ?, confirmed_by_count = ? "
                "WHERE id = ?",
                (json.dumps(confirmers), len(confirmers), row["id"]),
            )

    conn.commit()

    # Return the max confirmation count for this signature
    result = cursor.execute(
        "SELECT MAX(confirmed_by_count) FROM graph_edges "
        "WHERE source_table_key = ? AND source_column = ? "
        "AND target_table_key = ? AND target_column = ?",
        (source_table_key, source_column, target_table_key, target_column),
    ).fetchone()

    conn.close()
    return result[0] or 0


def get_promoted_suggestions(graph_id: str) -> List[StoredEdge]:
    """Get edges from other graphs that have reached the promotion threshold.

    Returns edges that:
    - Have >= PROMOTION_THRESHOLD unique confirmations across all graphs
    - Do not already exist in the target graph (by signature)
    """
    conn = _get_conn()

    # Get existing signatures in this graph
    existing = conn.execute(
        "SELECT source_table_key, source_column, target_table_key, target_column "
        "FROM graph_edges WHERE graph_id = ?",
        (graph_id,),
    ).fetchall()
    existing_sigs = {
        (r["source_table_key"], r["source_column"], r["target_table_key"], r["target_column"])
        for r in existing
    }

    # Find high-confidence edges from other graphs
    rows = conn.execute(
        "SELECT * FROM graph_edges "
        "WHERE graph_id != ? AND confirmed_by_count >= ?",
        (graph_id, PROMOTION_THRESHOLD),
    ).fetchall()

    promoted: List[StoredEdge] = []
    for row in rows:
        sig = (row["source_table_key"], row["source_column"],
               row["target_table_key"], row["target_column"])
        if sig not in existing_sigs:
            promoted.append(StoredEdge(
                id=row["id"], graph_id=row["graph_id"],
                source_table_key=row["source_table_key"],
                source_column=row["source_column"],
                target_table_key=row["target_table_key"],
                target_column=row["target_column"],
                edge_type=row["edge_type"], confidence=row["confidence"],
                rel_type=row["rel_type"], annotation=row["annotation"],
                evidence_json=row["evidence_json"], origin=row["origin"],
                confirmed_by_count=row["confirmed_by_count"],
                confirmers_json=row["confirmers_json"],
            ))

    conn.close()
    return promoted


# ---------------------------------------------------------------------------
# Import / Export helpers
# ---------------------------------------------------------------------------

def export_graph(graph_id: str) -> Dict[str, Any]:
    """Export a complete graph as a serializable dictionary."""
    graph = get_graph(graph_id)
    if graph is None:
        raise ValueError(f"Graph {graph_id} not found")

    nodes = list_nodes(graph_id)
    edges = list_edges(graph_id)

    return {
        "graph": asdict(graph),
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }


def import_graph(
    data: Dict[str, Any],
    new_id: Optional[str] = None,
    created_by: str = "",
) -> StoredGraph:
    """Import a graph from an exported dictionary.

    Args:
        data: Dictionary from export_graph() or similar shape.
        new_id: Override the graph ID (auto-generated if None).
        created_by: Creator identifier.

    Returns:
        The saved StoredGraph.
    """
    g_data = data.get("graph", {})
    graph = StoredGraph(
        id=new_id or g_data.get("id", str(uuid.uuid4())),
        name=g_data.get("name", ""),
        description=g_data.get("description", ""),
        domain_tag=g_data.get("domain_tag", ""),
        database_server=g_data.get("database_server", ""),
        created_by=created_by or g_data.get("created_by", ""),
        version=g_data.get("version", 1),
        is_template=bool(g_data.get("is_template", False)),
        parent_graph_id=g_data.get("parent_graph_id"),
    )
    save_graph(graph)

    for n_data in data.get("nodes", []):
        node = StoredNode(
            graph_id=graph.id,
            connection_id=n_data.get("connection_id", ""),
            schema_name=n_data.get("schema_name", ""),
            table_name=n_data.get("table_name", ""),
            row_count=n_data.get("row_count", 0),
            columns_json=json.dumps(n_data.get("columns", {})),
            descriptions_json=json.dumps(n_data.get("descriptions", {})),
        )
        save_node(node)

    for e_data in data.get("edges", []):
        edge = StoredEdge(
            graph_id=graph.id,
            source_table_key=e_data.get("source_table_key", ""),
            source_column=e_data.get("source_column", ""),
            target_table_key=e_data.get("target_table_key", ""),
            target_column=e_data.get("target_column", ""),
            edge_type=e_data.get("edge_type", "suggested"),
            confidence=e_data.get("confidence", 0.0),
            rel_type=e_data.get("rel_type", "one-to-many"),
            annotation=e_data.get("annotation", ""),
            evidence_json=json.dumps(e_data.get("evidence", {})),
            origin=e_data.get("origin", "user"),
            confirmed_by_count=e_data.get("confirmed_by_count", 0),
            confirmers_json=json.dumps(e_data.get("confirmers", [])),
        )
        save_edge(edge)

    return graph
