"""Assembles schema context for the SQL Generator agent.

Pulls nodes, edges, and column descriptions from a persisted graph and
formats them into a system prompt the LLM can use to generate SQL.
"""

import logging
from typing import Any, Dict, List, Optional

from src import graph_store
from src.chat.models import GeneratedQuery

logger = logging.getLogger(__name__)


def build_schema_context(
    graph_id: str,
    domain_context: str = "",
    max_tables: int = 30,
) -> Dict[str, Any]:
    """Build schema context from a persisted graph.

    Args:
        graph_id: ID of the persisted graph.
        domain_context: Optional domain description to inject.
        max_tables: Maximum number of tables to include in the prompt.

    Returns:
        Dictionary with keys:
            - graph: StoredGraph
            - nodes: list of StoredNode
            - edges: list of StoredEdge (confirmed + promoted suggestions)
            - prompt: str — assembled system prompt for the SQL Generator
    """
    graph = graph_store.get_graph(graph_id)
    if graph is None:
        raise ValueError(f"Graph {graph_id} not found")

    nodes = graph_store.list_nodes(graph_id)
    edges = graph_store.list_edges(graph_id)

    confirmed_edges = [e for e in edges if e.edge_type == "confirmed"]
    promoted = graph_store.get_promoted_suggestions(graph_id)
    promoted_edges = [
        e for e in promoted
        if e.source_table_key in {n.full_table_name for n in nodes}
        and e.target_table_key in {n.full_table_name for n in nodes}
    ]

    # Limit tables to avoid context blow-up
    selected_nodes = nodes[:max_tables]
    selected_keys = {n.full_table_name for n in selected_nodes}

    # Filter edges to selected tables only
    filtered_confirmed = [
        e for e in confirmed_edges
        if e.source_table_key in selected_keys and e.target_table_key in selected_keys
    ]
    filtered_promoted = [
        e for e in promoted_edges
        if e.source_table_key in selected_keys and e.target_table_key in selected_keys
    ]

    prompt = _build_prompt(
        graph=graph,
        nodes=selected_nodes,
        confirmed_edges=filtered_confirmed,
        promoted_edges=filtered_promoted,
        domain_context=domain_context,
        total_tables=len(nodes),
    )

    return {
        "graph": graph,
        "nodes": selected_nodes,
        "edges": filtered_confirmed,
        "promoted_edges": filtered_promoted,
        "prompt": prompt,
    }


def _build_prompt(
    graph: Any,
    nodes: List[Any],
    confirmed_edges: List[Any],
    promoted_edges: List[Any],
    domain_context: str,
    total_tables: int,
) -> str:
    """Assemble the system prompt string."""
    lines: List[str] = []

    lines.append(
        "DIALECT: Microsoft SQL Server (T-SQL). The target server is "
        f"{graph.database_server or 'unknown'}. "
        "You MUST generate queries that are syntactically valid for "
        "Microsoft SQL Server 2017 or later (T-SQL). Do NOT use syntax "
        "from other dialects such as MySQL, PostgreSQL, Oracle, or SQLite."
    )
    lines.append("")
    lines.append(
        "You are a SQL Server (T-SQL) query generator. Given a user's "
        "natural-language question and a schema description, produce a "
        "single SELECT statement that answers the question."
    )
    lines.append("")
    lines.append("RULES:")
    lines.append("1. Output ONLY a single SELECT statement. No DDL, DML, or EXEC.")
    lines.append("2. Use the schema exactly as provided — do not invent table or column names.")
    lines.append("3. Use JOINs based on the confirmed relationships listed below.")
    lines.append("4. Always add TOP (N) where N <= 1000 to avoid unbounded result sets.")
    lines.append("5. Wrap the response in ```sql ... ``` fences so it can be extracted.")
    lines.append("")
    lines.append("SQL SERVER SYNTAX GUIDE (use these, not generic SQL):")
    lines.append("- Row limiting: SELECT TOP (n) ...  (NOT LIMIT, NOT FETCH FIRST n ROWS ONLY)")
    lines.append("- Current timestamp: GETDATE() or SYSDATETIME()  (NOT CURRENT_TIMESTAMP in some cases, but it is supported)")
    lines.append("- Null handling: ISNULL(expr, replacement)  (NOT NVL)")
    lines.append("- String concat: + operator or CONCAT()  (NOT CONCAT() with ||)")
    lines.append("- Length: LEN() for character count, DATALENGTH() for bytes  (NOT LENGTH())")
    lines.append("- Date parts: DATEPART(YEAR, date), DATEPART(MONTH, date), YEAR(date), MONTH(date)")
    lines.append("- Date arithmetic: DATEADD(DAY, n, date), DATEDIFF(DAY, start, end)")
    lines.append("- Type casting: CAST(x AS INT) or CONVERT(INT, x)  (avoid CAST AS ::type)")
    lines.append("- Boolean: use expressions like = 1, <> 0 (SQL Server has no native BOOLEAN type)")
    lines.append("- Case-insensitive compare: use COLLATE Latin1_General_CI_AS or UPPER()/LOWER() on both sides")
    lines.append("- Quoting: identifiers use [brackets] or \"double quotes\"; string literals use 'single quotes'")
    lines.append("- Pagination: OFFSET n ROWS FETCH NEXT m ROWS ONLY (preferred over TOP for paging)")
    lines.append("- Aggregates: STRING_AGG(expr, ',')  (NOT GROUP_CONCAT)")
    lines.append("- Random: NEWID() inside ORDER BY for random sampling (NOT RAND())")
    lines.append("- Avoid: LIMIT, OFFSET without FETCH, ROWNUM, NVL, GROUP_CONCAT, ILIKE, :: cast, ? placeholders")
    lines.append("")

    if domain_context:
        lines.append(f"DOMAIN CONTEXT: {domain_context}")
        lines.append("")

    lines.append(f"GRAPH: {graph.name}")
    if graph.description:
        lines.append(f"DESCRIPTION: {graph.description}")
    if graph.database_server:
        lines.append(f"DATABASE SERVER: {graph.database_server}")
    lines.append(f"TABLES INCLUDED: {len(nodes)} of {total_tables}")
    lines.append("")

    lines.append("TABLES:")
    for node in nodes:
        cols = node.columns
        descs = node.descriptions
        lines.append(f"  {node.full_table_name} (~{node.row_count} rows)")
        for col_name, col_type in cols.items():
            desc = descs.get(col_name, "")
            if desc:
                lines.append(f"    - {col_name} ({col_type}): {desc}")
            else:
                lines.append(f"    - {col_name} ({col_type})")
    lines.append("")

    if confirmed_edges:
        lines.append("CONFIRMED RELATIONSHIPS:")
        for e in confirmed_edges:
            lines.append(
                f"  {e.source_table_key}.{e.source_column} -> "
                f"{e.target_table_key}.{e.target_column} ({e.rel_type})"
            )
        lines.append("")

    if promoted_edges:
        lines.append("PROMOTED SUGGESTIONS (high cross-user confidence):")
        for e in promoted_edges:
            lines.append(
                f"  {e.source_table_key}.{e.source_column} -> "
                f"{e.target_table_key}.{e.target_column} "
                f"(confirmed by {e.confirmed_by_count} users)"
            )
        lines.append("")

    return "\n".join(lines)


def extract_sql_from_response(text: str) -> Optional[str]:
    """Extract a SQL statement from an LLM response.

    Looks for ```sql ... ``` fences first, then falls back to looking for
    the first SELECT keyword.
    """
    text = text.strip()

    # Try code fence extraction
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i == 0:
                continue
            block = part
            # Strip language tag if present
            if block.startswith("sql"):
                block = block[3:].strip()
            elif block.startswith("tsql") or block.startswith("SQL"):
                block = block.split("\n", 1)[1].strip() if "\n" in block else block
            else:
                block = block.strip()
            if "SELECT" in block.upper():
                return block

    # Fallback: look for SELECT keyword
    upper = text.upper()
    idx = upper.find("SELECT")
    if idx == -1:
        return None
    return text[idx:].strip().rstrip(";").strip() or None


def validate_sql_safety(sql: str) -> Optional[str]:
    """Return an error message if the SQL is unsafe, else None.

    Blocks any non-SELECT statement (DDL, DML, EXEC, etc.).
    """
    if not sql:
        return "Empty SQL"

    cleaned = sql.strip().rstrip(";").strip()
    upper = cleaned.upper()

    if not upper.startswith("SELECT") and not upper.startswith("WITH"):
        return "Only SELECT queries are allowed"

    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER",
        "CREATE", "EXEC", "EXECUTE", "MERGE", "GRANT", "REVOKE",
    ]
    for kw in forbidden:
        # Check for word boundary
        if f" {kw} " in f" {upper} " or upper.startswith(kw):
            return f"Forbidden keyword: {kw}"

    return None


def parse_sql_generator_response(response: str) -> GeneratedQuery:
    """Parse the SQL Generator's response into a GeneratedQuery."""
    sql = extract_sql_from_response(response) or ""
    safety_error = validate_sql_safety(sql) if sql else "No SQL extracted"

    # Strip SQL from the response to get reasoning
    reasoning = response
    if "```" in response:
        parts = response.split("```")
        reasoning = parts[0].strip()
    elif sql:
        reasoning = response.replace(sql, "").strip()

    return GeneratedQuery(
        sql=sql,
        reasoning=reasoning,
        target_connection_id="",
    )
