"""Markdown, Mermaid, and JSON export for curated SQL relationships.

Generates:
- Structured markdown with frontmatter, table definitions, relationship matrix,
  evidence summaries, and Mermaid ER diagrams.
- JSON report with full machine-readable relationship set including all evidence.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.graph import RelationshipGraph, ConfirmedEdge, SuggestedEdge


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def generate_markdown(
    graph: RelationshipGraph,
    title: str = "SQL Relationships",
    databases: Optional[List[Dict[str, str]]] = None,
    include_evidence: bool = True,
    include_ai_descriptions: bool = False,
    ai_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
    include_suggested: bool = True,
) -> str:
    """Generate a complete markdown relationship document.

    Args:
        graph: The curated relationship graph.
        title: Title for the document.
        databases: List of connection metadata (server, database).
        include_evidence: Whether to include evidence summaries per relationship.
        include_suggested: Whether to include suggested (unconfirmed) relationships
            and their evidence summaries.

    Returns:
        Markdown string.
    """
    if databases is None:
        databases = []

    lines: List[str] = []
    lines.append("---")
    lines.append(f'title: "{title}"')
    lines.append(f"generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"table_count: {len(graph.nodes)}")
    lines.append(f"relationship_count: {len(graph.confirmed_edges)}")
    lines.append(f"database_count: {len(databases)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")

    # Source databases
    if databases:
        lines.append("## Source Databases")
        lines.append("")
        for db in databases:
            server = db.get("server", "?")
            database = db.get("database", "?")
            lines.append(f"- **Server:** `{server}`, **Database:** `{database}`")
        lines.append("")

    # Tables section
    lines.append("## Tables")
    lines.append("")

    for node_key in sorted(graph.nodes):
        attrs = graph.get_node(node_key)
        if attrs is None:
            continue

        schema = attrs.get("schema", "")
        table = attrs.get("table", "")
        database = attrs.get("database", "")
        row_count = attrs.get("row_count", 0)
        columns: Dict[str, str] = attrs.get("columns", {})
        primary_keys: List[str] = attrs.get("primary_keys", [])

        full_name = f"{database}.{schema}.{table}"
        annotation = graph.get_annotation(node_key)

        lines.append(f"### `{full_name}`")
        lines.append("")

        if annotation:
            lines.append(f"**Description:** {annotation}")
            lines.append("")

        lines.append(f"- **Rows:** {row_count:,}")
        lines.append(f"- **Columns:** {len(columns)}")

        if primary_keys:
            lines.append(f"- **Primary Keys:** `{'`, `'.join(primary_keys)}`")

        lines.append("")
        if include_ai_descriptions and ai_descriptions:
            table_descs = ai_descriptions.get(f"{schema}.{table}", {})
            lines.append("| Column | Type | Description |")
            lines.append("|--------|------|-------------|")
            for col_name, col_type in sorted(columns.items()):
                desc = table_descs.get(col_name, "")
                lines.append(f"| `{col_name}` | {col_type} | {desc} |")
        else:
            lines.append("| Column | Type |")
            lines.append("|--------|------|")
            for col_name, col_type in sorted(columns.items()):
                lines.append(f"| `{col_name}` | {col_type} |")
        lines.append("")

    # Relationships section
    lines.append("## Relationships")
    lines.append("")

    if graph.confirmed_edges:
        lines.append("| Source | Column | Target | Column | Type | Annotation |")
        lines.append("|--------|--------|--------|--------|------|------------|")
        for edge in graph.confirmed_edges:
            src_label = _short_label(edge.source_key)
            tgt_label = _short_label(edge.target_key)
            ann = edge.annotation.replace("|", "\\|") if edge.annotation else ""
            lines.append(
                f"| `{src_label}` | `{edge.source_column}` | "
                f"`{tgt_label}` | `{edge.target_column}` | "
                f"{edge.rel_type} | {ann} |"
            )
        lines.append("")
    else:
        lines.append("*No confirmed relationships.*")
        lines.append("")

    # Evidence summaries (for suggested edges that have evidence)
    if include_suggested and include_evidence and graph.suggested_edges:
        lines.append("## Evidence Summary (Suggested Relationships)")
        lines.append("")
        for edge in graph.suggested_edges:
            src_label = _short_label(edge.source_key)
            tgt_label = _short_label(edge.target_key)
            lines.append(f"### {src_label}.{edge.source_column} -> {tgt_label}.{edge.target_column}")
            lines.append("")
            lines.append(f"- **Confidence:** {edge.confidence:.4f} ({edge.confidence_band})")

            evidence = edge.evidence
            if evidence:
                lines.append(f"- **Name Score:** {evidence.get('name_score', 'N/A')}")
                lines.append(f"- **Type Score:** {evidence.get('type_score', 'N/A')}")
                lines.append(f"- **Value Score:** {evidence.get('value_score', 'N/A')}")
                lines.append(f"- **Uniqueness Score:** {evidence.get('uniqueness_score', 'N/A')}")
                lines.append(f"- **Null Score:** {evidence.get('null_score', 'N/A')}")
                lines.append(f"- **Match Type:** {evidence.get('match_type', 'N/A')}")
                lines.append(f"- **Reason:** {evidence.get('reason', 'N/A')}")

                ve = evidence.get("value_evidence")
                if ve:
                    lines.append(f"- **Jaccard:** {ve.get('jaccard', 'N/A')}")
                    lines.append(f"- **Overlap Ratio:** {ve.get('exact_overlap_ratio', 'N/A')}")
                    lines.append(f"- **Flags:** {', '.join(ve.get('confidence_flags', [])) or 'none'}")

                se = evidence.get("string_evidence")
                if se:
                    lines.append(f"- **Categorical Alignment:** {se.get('categorical_alignment', 'N/A')}")
                    lines.append(f"- **Token Similarity:** {se.get('token_similarity', 'N/A')}")

            lines.append("")

    # Suggested edges summary
    if include_suggested and graph.suggested_edges:
        lines.append("## Suggested Relationships (Pending Review)")
        lines.append("")
        lines.append("| Source | Column | Target | Column | Confidence | Band |")
        lines.append("|--------|--------|--------|--------|------------|------|")
        for edge in graph.suggested_edges:
            src_label = _short_label(edge.source_key)
            tgt_label = _short_label(edge.target_key)
            lines.append(
                f"| `{src_label}` | `{edge.source_column}` | "
                f"`{tgt_label}` | `{edge.target_column}` | "
                f"{edge.confidence:.2f} | {edge.confidence_band} |"
            )
        lines.append("")

    # Annotations section
    if graph.annotations:
        lines.append("## Annotations")
        lines.append("")
        for key, text in sorted(graph.annotations.items()):
            lines.append(f"- **`{key}`:** {text}")
        lines.append("")

    # Mermaid ER diagram
    lines.append("## Entity Relationship Diagram")
    lines.append("")
    lines.append("```mermaid")
    lines.append("erDiagram")

    # Entity declarations
    for node_key in sorted(graph.nodes):
        attrs = graph.get_node(node_key)
        if attrs is None:
            continue
        table = attrs.get("table", node_key.split(".")[-1])
        lines.append(f"    {table} {{")
        for col_name, col_type in (attrs.get("columns", {})).items():
            lines.append(f"        {col_type} {col_name}")
        lines.append(f"    }}")

    # Relationships
    for edge in graph.confirmed_edges:
        src_table = edge.source_key.split(".")[-1]
        tgt_table = edge.target_key.split(".")[-1]
        left_sym, right_sym = _rel_type_to_mermaid(edge.rel_type)
        label = f"\"{edge.source_column} -> {edge.target_column}\""
        lines.append(f"    {src_table} {left_sym}--{right_sym} {tgt_table} : {label}")

    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def generate_json_report(
    graph: RelationshipGraph,
    title: str = "SQL Relationships",
    databases: Optional[List[Dict[str, str]]] = None,
    include_ai_descriptions: bool = False,
    ai_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
    include_suggested: bool = True,
) -> Dict[str, Any]:
    """Generate a structured JSON relationship report.

    Includes table metadata, relationships with confidence scores,
    evidence, and annotations.

    Args:
        graph: The curated relationship graph.
        title: Title.
        databases: List of connection metadata.
        include_suggested: Whether to include suggested (unconfirmed) relationships.

    Returns:
        Serializable dictionary for JSON export.
    """
    if databases is None:
        databases = []

    # Table nodes
    tables: List[Dict[str, Any]] = []
    for node_key in sorted(graph.nodes):
        attrs = graph.get_node(node_key)
        if attrs is None:
            continue
        table_descs: Dict[str, str] = {}
        if include_ai_descriptions and ai_descriptions:
            table_descs = ai_descriptions.get(f"{attrs.get('schema', '')}.{attrs.get('table', '')}", {})

        table_entry: Dict[str, Any] = {
            "key": node_key,
            "database": attrs.get("database", ""),
            "schema": attrs.get("schema", ""),
            "table": attrs.get("table", ""),
            "row_count": attrs.get("row_count", 0),
            "columns": attrs.get("columns", {}),
            "primary_keys": attrs.get("primary_keys", []),
            "annotation": graph.get_annotation(node_key),
        }
        if include_ai_descriptions and table_descs:
            table_entry["column_descriptions"] = {
                col: table_descs[col]
                for col in attrs.get("columns", {})
                if col in table_descs
            }
        tables.append(table_entry)

    # Confirmed relationships
    relationships: List[Dict[str, Any]] = []
    for edge in graph.confirmed_edges:
        edge_key = f"{edge.source_key}->{edge.target_key}"
        relationships.append({
            "source_key": edge.source_key,
            "source_column": edge.source_column,
            "target_key": edge.target_key,
            "target_column": edge.target_column,
            "relationship_type": edge.rel_type,
            "annotation": edge.annotation,
            "edge_annotation": graph.get_annotation(edge_key),
        })

    # Suggested relationships with evidence
    suggested: List[Dict[str, Any]] = []
    if include_suggested:
        for edge in graph.suggested_edges:
            suggested.append({
                "source_key": edge.source_key,
                "source_column": edge.source_column,
                "target_key": edge.target_key,
                "target_column": edge.target_column,
                "confidence": edge.confidence,
                "confidence_band": edge.confidence_band,
                "evidence": edge.evidence,
            })

    return {
        "version": "1.0.0",
        "title": title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "databases": databases,
        "summary": {
            "table_count": len(graph.nodes),
            "confirmed_relationship_count": len(graph.confirmed_edges),
            "suggested_relationship_count": len(suggested),
        },
        "tables": tables,
        "relationships": relationships,
        "suggested_relationships": suggested,
        "annotations": dict(graph.annotations),
    }


def write_json_report(
    graph: RelationshipGraph,
    filepath: str,
    title: str = "SQL Relationships",
    databases: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Write a JSON relationship report to disk.

    Args:
        graph: The curated relationship graph.
        filepath: Output file path.
        title: Title.
        databases: List of connection metadata.
    """
    report = generate_json_report(graph, title, databases)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_label(node_key: str) -> str:
    """Extract a short schema.table label from a full node key.

    Args:
        node_key: Full key like "conn1.dbo.customers".

    Returns:
        Short label like "dbo.customers".
    """
    parts = node_key.split(".")
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


def _rel_type_to_mermaid(rel_type: str) -> Tuple[str, str]:
    """Map relationship type to Mermaid ER diagram symbols.

    Args:
        rel_type: one-to-one, one-to-many, many-to-one, many-to-many.

    Returns:
        Tuple of (left_symbol, right_symbol).
    """
    mapping = {
        "one-to-one": ("|o", "o|"),
        "one-to-many": ("|o", "o{"),
        "many-to-one": ("}o", "o|"),
        "many-to-many": ("}o", "o{"),
    }
    return mapping.get(rel_type, ("|o", "o{"))
