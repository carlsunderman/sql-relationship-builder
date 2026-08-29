"""Streamlit entry point for the SQL Relationship Builder.

Split-pane layout:
- Left panel (25%): Connections, table browser, profiles, save/load, export
- Right panel (75%): Interactive graph, relationship editor
"""

import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

logger = logging.getLogger(__name__)

from src.ai_analysis import (
    AIAnalysisResult,
    ai_relationships_to_suggested_edges,
    discover_relationships_batched,
    generate_column_descriptions_batched,
)
from src.db import (
    close_connection,
    connect_to_sql_server,
    discover_columns,
    discover_databases,
    discover_foreign_keys,
    discover_tables,
)
from src.drift import build_snapshot, detect_drift
from src.export import generate_json_report, generate_markdown, write_json_report
from src.graph import RelationshipGraph, SuggestedEdge
from src import graph_store as _graph_store
from src.chat.orchestrator import ChatOrchestrator, build_chat_messages, prune_history
from src.chat.models import ChatMessage, ChatTurn, MessageRole
from src.llm import (
    ALL_PROVIDERS,
    PROVIDER_AZURE,
    PROVIDER_LOCAL,
    LLMClient,
    LLMConfig,
)
from src.metadata import build_inventory, build_inventory_for_tables
from src.models import TableInfo, TableMetadata
from src.pipeline import AnalysisResult, run_analysis
from src.profiler import TableProfile, profile_tables
from src.state import build_save_data
from src.string_profiler import analyze_string_columns

# ---- Page config ----
st.set_page_config(
    page_title="SQL Graph Chat",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Initialize session state ----
if "connections" not in st.session_state:
    st.session_state.connections = {}
if "relationship_graph" not in st.session_state:
    st.session_state.relationship_graph = RelationshipGraph()
if "selected_tables" not in st.session_state:
    st.session_state.selected_tables = []
if "table_metadata" not in st.session_state:
    st.session_state.table_metadata = {}
if "table_profiles" not in st.session_state:
    st.session_state.table_profiles = {}
if "config" not in st.session_state:
    st.session_state.config = {}
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False
if "string_profiles" not in st.session_state:
    st.session_state.string_profiles = {}
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
# Load LLM config from admin DB.
if "llm_config" not in st.session_state:
    _llm_config_from_db = None
    try:
        _admin_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "admin"
        )
        if os.path.isdir(_admin_dir):
            import importlib.util as _importlib_util
            _cs_path = os.path.join(_admin_dir, "config_store.py")
            _spec = _importlib_util.spec_from_file_location("config_store", _cs_path)
            _config_store = _importlib_util.module_from_spec(_spec)
            _spec.loader.exec_module(_config_store)
            _config_store.init_db()
            _config_store.seed_from_env()
            _active = _config_store.get_active_config()
            if _active:
                _llm_config_from_db = _config_store.to_llm_config(_active)
            st.session_state._config_store = _config_store
    except Exception:
        pass

    if _llm_config_from_db:
        st.session_state.llm_config = LLMConfig.from_dict(_llm_config_from_db)
    else:
        st.error(
            "No AI provider config found. Create one in the admin panel "
            "(Settings → Provider Configs → New Config)."
        )
        st.stop()
if "domain_context" not in st.session_state:
    st.session_state.domain_context = ""
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "ai_analysis_run" not in st.session_state:
    st.session_state.ai_analysis_run = False
if "ai_status" not in st.session_state:
    st.session_state.ai_status = ""
if "save_json_bytes" not in st.session_state:
    st.session_state.save_json_bytes = b""
if "snapshot_bytes" not in st.session_state:
    st.session_state.snapshot_bytes = b""
if "suggested_selected" not in st.session_state:
    st.session_state.suggested_selected = set()
if "suggested_page" not in st.session_state:
    st.session_state.suggested_page = 1
if "suggested_page_size" not in st.session_state:
    st.session_state.suggested_page_size = 25
if "suggested_band_filter" not in st.session_state:
    st.session_state.suggested_band_filter = "All"
if "suggested_table_filter" not in st.session_state:
    st.session_state.suggested_table_filter = ""
if "suggested_source_filter" not in st.session_state:
    st.session_state.suggested_source_filter = "All"

# ---- Chat session state ----
if "chat_session_id" not in st.session_state:
    # Unique session identifier used for cross-user confirmation tracking.
    import uuid as _uuid
    st.session_state.chat_session_id = str(_uuid.uuid4())
if "active_chat_graph_id" not in st.session_state:
    st.session_state.active_chat_graph_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List[ChatTurn]
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # List[ChatMessage] (serialized for context)
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# Initialize the SQLite graph store on first run
_graph_store.init_db()


# ---- Config loading ----
def load_config() -> Dict[str, Any]:
    """Load configuration from defaults.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config", "defaults.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ====================================================================
# Helper Functions (defined before layout to avoid NameError on rerun)
# ====================================================================

# Brand palette
BRAND_BLUE = "#235BA8"
BRAND_NAVY = "#003070"
BRAND_RED = "#DF2027"


def _logo_data_uri() -> str:
    """Return the app logo as a base64 data URI (empty string if missing).

    Embedded inline so the header renders offline / in Kubernetes without
    hotlinking an external asset.
    """
    logo_path = os.path.join(
        os.path.dirname(__file__), "assets", "sql-ontology-builder-logo.png"
    )
    if not os.path.exists(logo_path):
        return ""
    with open(logo_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_header() -> None:
    """Render a full-width branded header with the app logo and colors."""
    logo_uri = _logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" alt="SQL Ontology Builder" style="height:54px;" />'
        if logo_uri
        else ""
    )
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:18px;
                    padding:12px 22px; background:#FFFFFF;
                    border-bottom:4px solid {BRAND_BLUE};
                    border-radius:6px; margin-bottom:10px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.08);">
            {logo_html}
            <div style="border-left:3px solid {BRAND_RED}; padding-left:18px;">
                <div style="font-size:1.55rem; font-weight:700;
                            color:{BRAND_NAVY}; line-height:1.15;">
                    SQL Relationship Builder
                </div>
                <div style="font-size:0.85rem; color:{BRAND_BLUE};
                            letter-spacing:0.04em;">
                    SQL Server Relationship Discovery
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_ai_description_lookup() -> Dict[str, Dict[str, str]]:
    """Build a lookup of AI column descriptions keyed by schema.table -> col -> description."""
    lookup: Dict[str, Dict[str, str]] = {}
    if st.session_state.ai_analysis_run and st.session_state.ai_result:
        for desc in st.session_state.ai_result.column_descriptions:
            lookup.setdefault(desc.table, {})[desc.column] = desc.description
    return lookup


def _load_selected_tables(all_tables: Dict[str, Dict[str, Any]]) -> None:
    """Load selected tables into the relationship graph and metadata store.

    Groups selected tables by connection and calls build_inventory_for_tables
    once per connection (4 bulk queries total per connection regardless of how
    many tables are selected) instead of the previous pattern of calling
    build_inventory per selected table, which scanned the entire schema and
    ran discover_columns + get_indexes for every table in it each time.
    """
    config = load_config()
    st.session_state.config = config
    graph = st.session_state.relationship_graph
    graph.graph.clear()
    graph.suggested_edges = []
    graph.confirmed_edges = []
    graph.annotations = {}
    st.session_state.table_metadata = {}

    # Group selected tables by connection to allow bulk metadata fetch.
    tables_by_conn: Dict[str, List[Dict[str, Any]]] = {}
    for table_key in st.session_state.selected_tables:
        table_data = all_tables.get(table_key)
        if table_data is None:
            continue
        conn_id = table_data["conn_id"]
        tables_by_conn.setdefault(conn_id, []).append(table_data)

    for conn_id, table_list in tables_by_conn.items():
        conn_info = st.session_state.connections[conn_id]
        conn = conn_info["connection"]

        schema_table_pairs = [
            (td["table_info"].schema_name, td["table_info"].table_name)
            for td in table_list
        ]

        # One set of bulk queries for all selected tables on this connection.
        # Fetch FKs once and reuse for both the inventory and the edge loop.
        all_fks = discover_foreign_keys(conn)
        inventory = build_inventory_for_tables(conn, schema_table_pairs, all_fks=all_fks)

        for td in table_list:
            table_info: TableInfo = td["table_info"]
            database: str = td["database"]
            meta_key = table_info.full_name
            metadata = inventory.get(meta_key)
            if metadata:
                table_info.columns = metadata.columns
                key = f"{conn_id}.{meta_key}"
                st.session_state.table_metadata[key] = metadata
                graph.add_table(conn_id, database, metadata)

        # Auto-add FK-based confirmed edges using the already-fetched FK list.
        for fk in all_fks:
            source_key = f"{conn_id}.{fk.source_schema}.{fk.source_table}"
            target_key = f"{conn_id}.{fk.target_schema}.{fk.target_table}"
            if source_key in graph.nodes and target_key in graph.nodes:
                try:
                    graph.confirm_edge(
                        source_key,
                        fk.source_column,
                        target_key,
                        fk.target_column,
                        rel_type="one-to-many",
                        annotation=f"FK: {fk.constraint_name}",
                    )
                except Exception:
                    pass

    # Auto-run profiling and string profiling
    _run_profiling()
    _run_string_profiling()


def _run_profiling() -> None:
    """Run profiling on loaded tables."""
    if not st.session_state.connections:
        return

    for conn_id, info in st.session_state.connections.items():
        conn = info["connection"]
        inventory: Dict[str, TableMetadata] = {}
        for key, meta in st.session_state.table_metadata.items():
            if key.startswith(conn_id):
                short_key = f"{meta.schema_name}.{meta.table_name}"
                inventory[short_key] = meta

        if inventory:
            profiles = profile_tables(conn, inventory, st.session_state.config)
            st.session_state.table_profiles = {}
            for short_key, profile in profiles.items():
                full_key = f"{conn_id}.{short_key}"
                st.session_state.table_profiles[full_key] = profile
        break


def _run_string_profiling() -> None:
    """Run string profiling on loaded tables."""
    if not st.session_state.connections:
        return

    for conn_id, info in st.session_state.connections.items():
        conn = info["connection"]
        inventory: Dict[str, TableMetadata] = {}
        for key, meta in st.session_state.table_metadata.items():
            if key.startswith(conn_id):
                short_key = f"{meta.schema_name}.{meta.table_name}"
                inventory[short_key] = meta

        if inventory and st.session_state.table_profiles:
            # Build short-key profiles dict
            short_profiles: Dict[str, TableProfile] = {}
            for key, profile in st.session_state.table_profiles.items():
                if key.startswith(conn_id):
                    short_key = key[len(conn_id) + 1 :]
                    short_profiles[short_key] = profile

            sp = analyze_string_columns(
                conn, short_profiles, inventory, st.session_state.config
            )
            # Map back to full keys
            st.session_state.string_profiles = {}
            for short_key, cols in sp.items():
                full_key = f"{conn_id}.{short_key}"
                st.session_state.string_profiles[full_key] = cols
        break


def _run_analysis_pipeline() -> None:
    """Run the full analysis pipeline to generate suggested edges."""
    graph = st.session_state.relationship_graph

    if not st.session_state.table_metadata:
        st.error("No tables loaded for analysis.")
        return

    if not st.session_state.table_profiles:
        st.error("Run profiling first before analysis.")
        return

    # Build the tables dict for the pipeline
    tables: Dict[str, TableMetadata] = {}
    profiles: Dict[str, TableProfile] = {}
    for key, meta in st.session_state.table_metadata.items():
        tables[key] = meta
        if key in st.session_state.table_profiles:
            profiles[key] = st.session_state.table_profiles[key]

    # Run the pipeline
    result = run_analysis(
        tables=tables,
        table_profiles=profiles,
        string_profiles=st.session_state.string_profiles,
        config=st.session_state.config,
    )
    st.session_state.analysis_result = result
    st.session_state.analysis_run = True

    # Load suggested edges into the graph
    graph.suggested_edges = []
    for edge_data in result.suggested_edges:
        # Map table keys to graph node keys
        src_table = edge_data["source_table"]
        tgt_table = edge_data["target_table"]

        # Find the graph node keys for these tables
        src_key = None
        tgt_key = None
        for node_key in graph.nodes:
            if node_key.endswith(src_table):
                src_key = node_key
            if node_key.endswith(tgt_table):
                tgt_key = node_key

        if src_key and tgt_key:
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


def _sync_edges_from_df(edited_df: pd.DataFrame) -> None:
    """Sync edited relationship table back to the relationship graph.

    Args:
        edited_df: Edited DataFrame from st.data_editor.
    """
    graph = st.session_state.relationship_graph

    # Build lookup by short name -> full key
    node_lookup: Dict[str, str] = {}
    for node_key in graph.nodes:
        short = node_key.split(".")[-1]
        node_lookup[short] = node_key

    # Remove all confirmed edges and rebuild from DataFrame
    graph.confirmed_edges = []

    for _, row in edited_df.iterrows():
        source_short = str(row.get("Source", ""))
        target_short = str(row.get("Target", ""))
        source_col = str(row.get("Source Column", ""))
        target_col = str(row.get("Target Column", ""))
        rel_type = str(row.get("Type", "one-to-many"))
        annotation = str(row.get("Annotation", ""))

        src_key = node_lookup.get(source_short)
        tgt_key = node_lookup.get(target_short)

        if src_key and tgt_key:
            graph.confirm_edge(
                src_key,
                source_col,
                tgt_key,
                target_col,
                rel_type=rel_type,
                annotation=annotation,
            )


def _render_saved_connection_form(
    saved_connections: List[Any],
) -> None:
    """Render the saved-connection form with server + database dropdowns."""
    with st.form("saved_connection_form", clear_on_submit=False):
        conn_options = [sc.id for sc in saved_connections]
        conn_labels = [f"{sc.name} ({sc.server})" for sc in saved_connections]

        selected_id = st.selectbox(
            "Server",
            options=conn_options,
            format_func=lambda oid: conn_labels[conn_options.index(oid)],
            key="saved_server_select",
        )

        # Resolve the selected connection
        selected_sc = None
        for sc in saved_connections:
            if sc.id == selected_id:
                selected_sc = sc
                break

        # Build database options
        database_options: List[str] = []
        if selected_sc:
            allowed = json.loads(selected_sc.allowed_databases or "[]")
            if allowed:
                database_options = allowed
            else:
                # Discover databases at runtime
                try:
                    pw = ""
                    _cs = st.session_state.get("_config_store")
                    if _cs:
                        pw = _cs.get_connection_password(selected_sc)
                    database_options = discover_databases(
                        selected_sc.server,
                        selected_sc.username,
                        pw,
                    )
                except Exception as e:
                    st.error(f"Could not discover databases: {e}")

        database = st.selectbox(
            "Database",
            options=database_options,
            key="saved_database_select",
            help="Select a database. Options are populated from the saved connection.",
        )

        submitted = st.form_submit_button(
            "Connect", use_container_width=True, type="primary"
        )

    if submitted and selected_sc:
        if not database:
            st.error("Select a database.")
        else:
            try:
                pw = ""
                _cs = st.session_state.get("_config_store")
                if _cs:
                    pw = _cs.get_connection_password(selected_sc)
                conn = connect_to_sql_server(
                    selected_sc.server,
                    database,
                    selected_sc.username,
                    pw,
                )
                conn_id = f"{selected_sc.server}\\{database}"
                st.session_state.connections[conn_id] = {
                    "connection": conn,
                    "server": selected_sc.server,
                    "database": database,
                }
                st.success(f"Connected to {conn_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")


def _render_manual_connection_form() -> None:
    """Render the manual connection form with free-text inputs."""
    with st.form("manual_connection_form", clear_on_submit=False):
        server = st.text_input("Server", placeholder="localhost")
        database = st.text_input("Database")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Connect", use_container_width=True, type="primary"
        )

    if submitted:
        if not server or not database:
            st.error("Server and database are required.")
        else:
            try:
                conn = connect_to_sql_server(server, database, username, password)
                conn_id = f"{server}\\{database}"
                st.session_state.connections[conn_id] = {
                    "connection": conn,
                    "server": server,
                    "database": database,
                }
                st.success(f"Connected to {conn_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")


# ====================================================================
# Chat tab renderer
# ====================================================================

def _save_current_graph_to_store(name: str, description: str) -> None:
    """Persist the current session graph to the SQLite store."""
    graph = _graph_store.StoredGraph(
        name=name,
        description=description,
        domain_tag="",
        database_server=_active_database_server(),
        created_by="local",
        is_template=False,
    )
    _graph_store.save_graph(graph)

    # Build AI descriptions lookup once (keyed by schema.table -> col -> desc)
    ai_lookup = _build_ai_description_lookup()

    # Save nodes. node_key format: {conn_id}.{schema}.{table}
    for node_key in st.session_state.relationship_graph.nodes:
        node_data = st.session_state.relationship_graph.get_node(node_key) or {}
        parts = node_key.split(".")
        # parts = [conn_id, schema, table]
        conn_id = parts[0] if len(parts) >= 3 else ""
        schema_name = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) == 2 else "")
        table_name = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else "")

        columns = node_data.get("columns", {})
        # Look up descriptions by short schema.table key
        short_key = f"{schema_name}.{table_name}"
        descriptions = ai_lookup.get(short_key, {})

        stored = _graph_store.StoredNode(
            graph_id=graph.id,
            connection_id=conn_id,
            schema_name=schema_name,
            table_name=table_name,
            row_count=int(node_data.get("row_count", 0)),
            columns_json=json.dumps(columns),
            descriptions_json=json.dumps(descriptions),
        )
        _graph_store.save_node(stored)

    # Save confirmed edges
    for edge in st.session_state.relationship_graph.confirmed_edges:
        stored = _graph_store.StoredEdge(
            graph_id=graph.id,
            source_table_key=edge.source_key,
            source_column=edge.source_column,
            target_table_key=edge.target_key,
            target_column=edge.target_column,
            edge_type="confirmed",
            confidence=1.0,
            rel_type=edge.rel_type,
            annotation=edge.annotation or "",
            evidence_json=json.dumps(getattr(edge, "evidence", {}) or {}),
            origin="user",
        )
        _graph_store.save_edge(stored)

    st.session_state.active_chat_graph_id = graph.id


def _active_database_server() -> str:
    """Return a server\\database identifier for the first active connection."""
    if not st.session_state.connections:
        return ""
    first = next(iter(st.session_state.connections.values()))
    return f"{first.get('server', '')}\\{first.get('database', '')}"



def _auto_connect_for_graph(graph_db_server: str) -> Optional[Any]:
    """Auto-connect to a saved server using stored credentials.

    Args:
        graph_db_server: The server\\database identifier from the persisted graph.

    Returns:
        An open pyodbc connection, or None if auto-connection fails.
    """
    _cs = st.session_state.get("_config_store")
    if not _cs or "\\" not in graph_db_server:
        return None

    server, database = graph_db_server.split("\\", 1)

    # Check if already connected
    for info in st.session_state.connections.values():
        if info.get("server") == server and info.get("database") == database:
            return info.get("connection")

    # Find a saved connection matching this server
    saved = _cs.list_server_connections()
    for sc in saved:
        if sc.get("server") == server:
            try:
                conn = connect_to_sql_server(
                    server=server,
                    database=database,
                    username=sc.get("username", ""),
                    password=_cs.decrypt_password(sc.get("password_encrypted", "")),
                )
                conn_id = f"{server}_{database}"
                st.session_state.connections[conn_id] = {
                    "server": server,
                    "database": database,
                    "connection": conn,
                }
                return conn
            except Exception as e:
                logger.error("Auto-connect failed: %s", e)
                return None
    return None


def _render_graph_store_expander() -> None:
    """Render the graph store management UI in the left panel."""
    with st.expander("Saved Graphs", expanded=False):
        graphs = _graph_store.list_graphs()
        if not graphs:
            st.caption("No saved graphs yet.")
        else:
            options = {g.name + (" (template)" if g.is_template else ""): g.id for g in graphs}
            selected_label = st.selectbox(
                "Load graph",
                options=list(options.keys()),
                key="graph_store_selector",
            )
            if selected_label and st.button("Load", use_container_width=True):
                st.session_state.active_chat_graph_id = options[selected_label]
                st.rerun()

        with st.form("save_graph_form", clear_on_submit=False):
            save_name = st.text_input("Graph name", key="save_graph_name")
            save_desc = st.text_input("Description (optional)", key="save_graph_desc")
            save_submitted = st.form_submit_button(
                "Save current graph", use_container_width=True
            )

        if save_submitted and save_name:
            if not st.session_state.relationship_graph.nodes:
                st.error("No graph to save — load tables first.")
            else:
                _save_current_graph_to_store(save_name, save_desc)
                st.success(f"Saved '{save_name}'")
                st.rerun()

        # Fork current active graph
        active_id = st.session_state.active_chat_graph_id
        if active_id:
            active = _graph_store.get_graph(active_id)
            if active and not active.is_template:
                with st.form("fork_graph_form", clear_on_submit=False):
                    fork_name = st.text_input("Fork name", key="fork_graph_name")
                    fork_submitted = st.form_submit_button(
                        "Fork this graph", use_container_width=True
                    )
                if fork_submitted and fork_name:
                    try:
                        _graph_store.fork_graph(
                            source_id=active_id,
                            name=fork_name,
                            created_by="local",
                        )
                        st.success(f"Forked '{active.name}' as '{fork_name}'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fork failed: {e}")


def _render_chat_tab() -> None:
    """Render the chat interface tab."""
    st.subheader("Chat with your data")

    if not st.session_state.llm_config.is_configured():
        st.warning("Configure an LLM provider in the left panel first.")
        return

    # Graph selector
    graphs = _graph_store.list_graphs()
    if not graphs:
        st.info(
            "No saved graphs available. Save the current graph from the "
            "'Saved Graphs' expander in the left panel to enable chat."
        )
        return

    graph_labels = {
        (g.name + (" (template)" if g.is_template else "")): g.id for g in graphs
    }
    selected_label = st.selectbox(
        "Active graph",
        options=list(graph_labels.keys()),
        index=0,
        key="chat_graph_selector",
    )
    if selected_label:
        st.session_state.active_chat_graph_id = graph_labels[selected_label]

    if not st.session_state.active_chat_graph_id:
        return

    active_graph = _graph_store.get_graph(st.session_state.active_chat_graph_id)
    if active_graph is None:
        st.error("Selected graph no longer exists.")
        return

    # Connection selector — auto-connect if needed
    conn = _get_or_auto_connect(active_graph.database_server)
    if conn is None:
        st.warning(
            f"No connection available for {active_graph.database_server}. "
            "Connect from the left panel or save a server connection in admin."
        )
        return

    # Chat control buttons (pinned at top)
    _render_chat_controls(active_graph)

    # Input field (locked directly below controls so it doesn't get lost)
    _render_chat_input(active_graph, conn)

    # Render conversation history (newest first)
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown(f"**Conversation** ({len(st.session_state.chat_history)} turn{'s' if len(st.session_state.chat_history) != 1 else ''})")
        # Reverse so newest appears at top of the conversation list
        for idx, turn in enumerate(reversed(st.session_state.chat_history)):
            turn_num = len(st.session_state.chat_history) - idx
            _render_chat_turn(turn, turn_num)
    else:
        st.caption("No messages yet — ask a question above to get started.")


def _render_chat_controls(active_graph: Any) -> None:
    """Render the New Chat / Clear / Export buttons for the active conversation."""
    turn_count = len(st.session_state.chat_history)
    cols = st.columns([1, 1, 1, 4])
    with cols[0]:
        if st.button(
            "New Chat",
            use_container_width=True,
            disabled=(turn_count == 0),
            help="Start a fresh conversation (clears history).",
        ):
            st.session_state.chat_history = []
            st.session_state.chat_messages = []
            st.rerun()
    with cols[1]:
        if st.button(
            "Clear",
            use_container_width=True,
            disabled=(turn_count == 0),
            help="Clear all messages from the current conversation.",
        ):
            st.session_state.chat_history = []
            st.session_state.chat_messages = []
            st.rerun()
    with cols[2]:
        if st.button(
            "Export",
            use_container_width=True,
            disabled=(turn_count == 0),
            help="Download the conversation as JSON.",
        ):
            _export_chat_history(active_graph)
    with cols[3]:
        st.caption(
            f"{turn_count} turn{'s' if turn_count != 1 else ''} | "
            f"Graph: {active_graph.name}"
        )


def _export_chat_history(active_graph: Any) -> None:
    """Build a JSON download of the current conversation and surface it via Streamlit."""
    from dataclasses import asdict

    payload = {
        "graph_id": active_graph.id,
        "graph_name": active_graph.name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "session_id": st.session_state.get("chat_session_id", ""),
        "turns": [
            {
                "user_question": t.user_question,
                "generated_sql": t.generated_sql,
                "sql_reasoning": t.sql_reasoning,
                "answer": t.answer,
                "error": t.error,
                "timestamp": t.timestamp,
                "result": asdict(t.query_result) if t.query_result else None,
            }
            for t in st.session_state.chat_history
        ],
    }
    st.download_button(
        "Download conversation JSON",
        data=json.dumps(payload, indent=2, default=str),
        file_name=f"chat_{active_graph.name.replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True,
        key=f"chat_export_{len(st.session_state.chat_history)}",
    )


def _render_chat_input(active_graph: Any, conn: Any) -> None:
    """Render the pinned input area under the chat controls."""
    with st.container(border=True):
        st.markdown("**Ask a question**")
        # Use a form so Enter submits and the field doesn't reset mid-render
        with st.form("chat_input_form", clear_on_submit=True):
            user_input = st.text_area(
                "Question",
                placeholder="e.g., How many wells were drilled per operator in 2024?",
                height=80,
                label_visibility="collapsed",
                key="chat_input_text",
            )
            submit_cols = st.columns([1, 5])
            with submit_cols[0]:
                submitted = st.form_submit_button(
                    "Send", use_container_width=True, type="primary"
                )
            with submit_cols[1]:
                st.caption("Press Ctrl+Enter to send")
        if submitted and user_input and user_input.strip():
            _run_chat_turn(user_input.strip(), active_graph, conn)


def _render_chat_turn(turn: Any, turn_num: int) -> None:
    """Render a single conversation turn with user question + assistant response."""
    with st.container(border=True):
        # Header row with turn number
        ts_label = ""
        if turn.timestamp:
            try:
                ts_label = datetime.fromisoformat(turn.timestamp).strftime("%H:%M:%S")
            except (ValueError, TypeError):
                ts_label = ""
        header = f"**Turn {turn_num}**"
        if ts_label:
            header += f" &middot; _{ts_label}_"
        st.markdown(header)

        # User question
        st.markdown(f"**You:** {turn.user_question}")

        # Assistant response
        if turn.error:
            st.error(turn.error)
        if turn.generated_sql:
            with st.expander("View SQL", expanded=False):
                st.code(turn.generated_sql, language="sql")
        if turn.query_result and turn.query_result.rows:
            st.dataframe(
                pd.DataFrame(
                    turn.query_result.rows,
                    columns=turn.query_result.columns,
                ),
                use_container_width=True,
            )
        if turn.answer:
            st.markdown("**Answer:**")
            st.write(turn.answer)


def _get_or_auto_connect(database_server: str) -> Optional[Any]:
    """Get an active connection matching the graph's server, auto-connect if possible."""
    if not database_server or "\\" not in database_server:
        # Fall back to any active connection
        if st.session_state.connections:
            return next(iter(st.session_state.connections.values())).get("connection")
        return None

    server, database = database_server.split("\\", 1)

    # Check existing connections
    for conn_id, info in st.session_state.connections.items():
        if info.get("server") == server and info.get("database") == database:
            return info.get("connection")

    # Try auto-connect
    return _auto_connect_for_graph(database_server)


def _run_chat_turn(question: str, active_graph: Any, conn: Any) -> None:
    """Execute one chat turn and append to history."""
    orchestrator = ChatOrchestrator(
        llm=LLMClient(st.session_state.llm_config),
        row_limit=1000,
        query_timeout=30,
    )

    history_msgs = prune_history(st.session_state.chat_messages, max_turns=5)

    turn: ChatTurn = orchestrator.ask(
        question=question,
        graph_id=active_graph.id,
        connection=conn,
        domain_context=st.session_state.get("domain_context", ""),
        history=history_msgs,
    )

    st.session_state.chat_history.append(turn)
    st.session_state.chat_messages = prune_history(
        build_chat_messages(st.session_state.chat_history),
        max_turns=5,
    )
    st.rerun()

# ---- Layout ----
_render_header()

left_col, right_col = st.columns([0.25, 0.75], gap="medium")

with left_col:
    # ==============================
    # Connection Manager
    # ==============================
    with st.expander("Connections", expanded=True):
        # Load server connections from admin DB
        _saved_connections: List[Any] = []
        _cs_module = st.session_state.get("_config_store")
        if _cs_module:
            try:
                _saved_connections = _cs_module.list_server_connections()
            except Exception:
                pass

        # Toggle between saved and manual connection modes
        if "connection_mode" not in st.session_state:
            st.session_state.connection_mode = "saved" if _saved_connections else "manual"

        if _saved_connections:
            conn_mode = st.radio(
                "Connection type",
                options=["saved", "manual"],
                format_func=lambda m: "Saved Connection" if m == "saved" else "Manual",
                horizontal=True,
                key="conn_mode_radio",
                index=0 if st.session_state.connection_mode == "saved" else 1,
            )
            st.session_state.connection_mode = conn_mode
        else:
            conn_mode = "manual"

        if conn_mode == "saved":
            _render_saved_connection_form(_saved_connections)
        else:
            _render_manual_connection_form()

        # Active connections
        if st.session_state.connections:
            st.markdown("**Active Connections**")
            for conn_id, info in list(st.session_state.connections.items()):
                col_a, col_c = st.columns([5, 1])
                with col_a:
                    st.caption(f"{info['server']}/{info['database']}")
                with col_c:
                    if st.button("", key=f"dc_{conn_id}", help=f"Disconnect {conn_id}"):
                        close_connection(info.get("connection"))
                        del st.session_state.connections[conn_id]
                        st.rerun()

    # ==============================
    # Table Browser
    # ==============================
    with st.expander("Tables", expanded=True):
        all_tables: Dict[str, Dict[str, Any]] = {}

        for conn_id, info in st.session_state.connections.items():
            try:
                tables = discover_tables(info["connection"])
                for key, table_info in tables.items():
                    full_key = f"{conn_id}.{key}"
                    all_tables[full_key] = {
                        "conn_id": conn_id,
                        "database": info["database"],
                        "table_info": table_info,
                    }
            except Exception as e:
                st.error(f"Failed to discover tables: {e}")

        if all_tables:
            # Group tables by connection
            table_options = list(all_tables.keys())
            # Create display labels
            display_labels = {}
            for key in table_options:
                parts = key.split(".")
                if len(parts) >= 2:
                    display_labels[key] = f"{parts[-2]}.{parts[-1]}"
                else:
                    display_labels[key] = key

            selected = st.multiselect(
                "Select tables to load",
                options=table_options,
                format_func=lambda x: display_labels.get(x, x),
                default=st.session_state.selected_tables,
                key="table_selector",
            )
            st.session_state.selected_tables = selected

            if st.button(
                "Load Selected Tables", use_container_width=True, type="primary"
            ):
                _load_selected_tables(all_tables)
                st.rerun()

    # ==============================
    # Analysis
    # ==============================
    with st.expander("Analysis", expanded=False):
        if st.session_state.table_metadata:
            if st.button("Run Profiling", use_container_width=True):
                _run_profiling()
                st.rerun()

            if st.session_state.table_profiles:
                st.caption(f"Profiled {len(st.session_state.table_profiles)} tables")

            if st.button(
                "Run Analysis Pipeline", use_container_width=True, type="primary"
            ):
                _run_analysis_pipeline()
                st.rerun()

            if st.session_state.analysis_run:
                st.caption(
                    f"{len(st.session_state.relationship_graph.suggested_edges)} candidate(s) found"
                )
        else:
            st.caption("Load tables first")

    # ==============================
    # AI Configuration
    # ==============================
    with st.expander("AI Configuration", expanded=False):
        llm = st.session_state.llm_config

        provider = st.selectbox(
            "Provider",
            options=ALL_PROVIDERS,
            format_func=lambda p: {
                PROVIDER_AZURE: "Azure OpenAI",
                PROVIDER_LOCAL: "Local (Ollama / LM Studio / vLLM)",
            }.get(p, p),
            index=ALL_PROVIDERS.index(llm.provider),
            key="ai_provider",
        )
        llm.provider = provider

        if "last_ai_provider" not in st.session_state:
            st.session_state.last_ai_provider = provider

        if provider != st.session_state.last_ai_provider:
            _cs = st.session_state.get("_config_store")
            if _cs:
                _matching = [c for c in _cs.list_configs() if c.provider == provider]
                if _matching:
                    _chosen = max(_matching, key=lambda c: c.is_active)
                    _llm_dict = _cs.to_llm_config(_chosen)
                    llm.provider = _llm_dict["provider"]
                    llm.endpoint = _llm_dict["endpoint"]
                    llm.model = _llm_dict["model"]
                    llm.api_key = _llm_dict["api_key"]
                    llm.api_version = _llm_dict["api_version"]
                    llm.deployment_name = _llm_dict["deployment_name"]
                    llm.temperature = _llm_dict["temperature"]
                    llm.max_tokens = _llm_dict["max_tokens"]
                    llm.timeout = _llm_dict["timeout"]
                    llm.verify_ssl = _llm_dict["verify_ssl"]
                    st.session_state.ai_temperature = llm.temperature
                    st.session_state.ai_max_tokens = llm.max_tokens
                    st.session_state.ai_timeout = llm.timeout
                else:
                    st.warning(
                        f"No config found for {provider}. "
                        "Create one in the admin panel (Settings → Provider Configs → New Config)."
                    )

            st.session_state.ai_deployment = llm.deployment_name
            st.session_state.ai_api_version = llm.api_version
            st.session_state.last_ai_provider = provider
        if provider == PROVIDER_AZURE:
            st.markdown(f"**Azure Endpoint:** {llm.endpoint}")
            st.markdown(f"**Deployment Name:** {llm.deployment_name}")
        else:
            st.markdown(f"**API Endpoint:** {llm.endpoint}")

        st.markdown(f"**Model:** {llm.model}")
        # API key is provided via environment/session, not entered in the UI.
        key_present = bool((llm.api_key or "").strip())
        if provider == PROVIDER_AZURE:
            st.caption(
                "API Key: ✅ Provided via environment/session"
                if key_present
                else "API Key: ❌ Missing (set env var)"
            )
        else:
            st.caption(
                "API Key: ✅ Provided via environment/session"
                if key_present
                else "API Key: optional for local provider"
            )
        if provider == PROVIDER_AZURE:
            st.markdown(f"**API Version:** {llm.api_version}")

            # Seed safer defaults on first Azure render (without overriding existing choices)
            st.session_state.setdefault("ai_temperature", 0.0)
            st.session_state.setdefault("ai_max_tokens", 1024)
            st.session_state.setdefault("ai_batch_size", 2)
            st.session_state.setdefault("ai_timeout", 300)

            st.caption(
                "Azure recommended defaults: Temperature 0.0, Max Tokens 1024, "
                "Batch Size 2, Timeout 300s."
            )

        col_temp, col_tokens, col_batch = st.columns(3)
        with col_temp:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=llm.temperature,
                step=0.1,
                key="ai_temperature",
            )
            llm.temperature = temperature
        with col_tokens:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=256,
                max_value=128000,
                value=llm.max_tokens,
                step=256,
                key="ai_max_tokens",
            )
            llm.max_tokens = max_tokens
        with col_batch:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("ai_batch_size", 8)),
                key="ai_batch_size",
                help="Tables per LLM call. Lower for smaller context windows.",
            )

        timeout_val = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=600,
            value=llm.timeout,
            step=10,
            key="ai_timeout",
        )
        llm.timeout = timeout_val

        verify_ssl = st.checkbox(
            "Verify SSL certificates",
            value=llm.verify_ssl,
            key="ai_verify_ssl",
            help="Uncheck if connections fail with SSL certificate errors (e.g., corporate proxy with self-signed cert).",
        )
        llm.verify_ssl = verify_ssl

        if llm.is_configured():
            st.success("AI configured")
        else:
            missing = []
            if not llm.model:
                missing.append("model")
            if provider == PROVIDER_AZURE:
                if not llm.endpoint:
                    missing.append("endpoint")
                if not llm.api_key:
                    missing.append("api key")
                if not llm.deployment_name:
                    missing.append("deployment name")
            else:
                if not llm.endpoint:
                    missing.append("endpoint")
            if missing:
                st.warning(f"Missing: {', '.join(missing)}")

    # ==============================
    # AI Analysis
    # ==============================
    with st.expander("AI Analysis", expanded=False):
        st.caption(
            "Use an LLM to generate column descriptions and discover hidden relationships."
        )

        domain_context = st.text_area(
            "Domain Context",
            value=st.session_state.domain_context,
            height=100,
            placeholder=(
                "Describe your data domain. Example:\n\n"
                "This data is related to Oil and Gas Exploration and Production, "
                "sourced from BOEM and other government agencies for use by private "
                "companies such as Shell, BP, etc."
            ),
            key="domain_context_input",
            help=(
                "Provide context about your data domain to help the LLM generate "
                "more accurate descriptions and identify domain-specific relationships."
            ),
        )
        st.session_state.domain_context = domain_context

        if st.session_state.table_metadata:
            if st.button(
                "Run AI Analysis",
                use_container_width=True,
                type="primary",
                key="run_ai_analysis",
            ):
                cfg = st.session_state.llm_config
                if not cfg.is_configured():
                    missing = []
                    if not cfg.model:
                        missing.append("model")
                    if not cfg.endpoint:
                        missing.append("endpoint")
                    if cfg.provider == PROVIDER_AZURE:
                        if not cfg.api_key:
                            missing.append("api key")
                        if not cfg.deployment_name:
                            missing.append("deployment name")
                    detail = f" Missing: {', '.join(missing)}." if missing else ""
                    st.error(
                        "Configure AI settings first (AI Configuration panel)."
                        + detail
                    )
                else:
                    progress = st.progress(0, text="Starting AI analysis...")
                    try:
                        client = LLMClient(st.session_state.llm_config)

                        tables_short: Dict[str, TableMetadata] = {}
                        profiles_short: Dict[str, TableProfile] = {}
                        for key, meta in st.session_state.table_metadata.items():
                            short_key = f"{meta.schema_name}.{meta.table_name}"
                            tables_short[short_key] = meta
                            full_profile_key = key
                            if full_profile_key in st.session_state.table_profiles:
                                profiles_short[short_key] = (
                                    st.session_state.table_profiles[full_profile_key]
                                )

                        existing_rels = []
                        for edge in st.session_state.relationship_graph.confirmed_edges:
                            existing_rels.append(
                                {
                                    "source": edge.source_key.split(".")[-1],
                                    "source_col": edge.source_column,
                                    "target": edge.target_key.split(".")[-1],
                                    "target_col": edge.target_column,
                                }
                            )

                        batch_size = int(st.session_state.get("ai_batch_size", 8))

                        progress.progress(10, text="Generating column descriptions...")
                        descs, desc_errs = generate_column_descriptions_batched(
                            client,
                            tables_short,
                            profiles_short,
                            domain_context,
                            batch_size=batch_size,
                            timeout=st.session_state.llm_config.timeout,
                        )
                        progress.progress(50, text="Discovering relationships...")
                        rels, rel_errs = discover_relationships_batched(
                            client,
                            tables_short,
                            profiles_short,
                            domain_context,
                            existing_rels,
                            batch_size=batch_size,
                            timeout=st.session_state.llm_config.timeout,
                        )

                        ai_result = AIAnalysisResult(
                            column_descriptions=descs,
                            relationships=rels,
                            errors=desc_errs + rel_errs,
                        )
                        st.session_state.ai_result = ai_result
                        st.session_state.ai_analysis_run = True

                        if ai_result.relationships:
                            node_lookup: Dict[str, str] = {}
                            for node_key in st.session_state.relationship_graph.nodes:
                                parts = node_key.split(".")
                                if len(parts) >= 2:
                                    short = f"{parts[-2]}.{parts[-1]}"
                                    node_lookup[short] = node_key

                            ai_edges = ai_relationships_to_suggested_edges(
                                ai_result, node_lookup
                            )
                            for edge in ai_edges:
                                st.session_state.relationship_graph.add_suggested_edge(edge)

                        progress.progress(100, text="Complete")
                        st.rerun()

                    except ImportError as e:
                        st.error(
                            f"Missing dependency: {e}. Install with: uv add openai"
                        )
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")

        # Show status summary
        if st.session_state.ai_analysis_run and st.session_state.ai_result:
            ai_res = st.session_state.ai_result
            parts = [
                f"{len(ai_res.column_descriptions)} descriptions",
                f"{len(ai_res.relationships)} relationships",
            ]
            if ai_res.errors:
                parts.append(f"{len(ai_res.errors)} partial error(s)")
                st.warning(" ".join(parts))
                show_errors = st.checkbox("Show errors", key="ai_show_errors")
                if show_errors:
                    for err in ai_res.errors:
                        st.caption(f"- {err}")
            else:
                st.success(" ".join(parts))
        elif not st.session_state.table_metadata:
            st.caption("Load tables first")

    # ==============================
    # Save / Load
    # ==============================
    with st.expander("Save / Load", expanded=False):
        st.markdown("**Relationships JSON Save / Load**")

        save_name = st.text_input("Filename", value="relationships.json", key="save_name")
        connections_meta = [
            {"server": info["server"], "database": info["database"]}
            for info in st.session_state.connections.values()
        ]
        save_data = build_save_data(
            connections_meta,
            st.session_state.selected_tables,
            st.session_state.relationship_graph.to_dict(),
        )

        if st.button("Prepare download", use_container_width=True):
            st.session_state.save_json_bytes = json.dumps(
                save_data, indent=2, default=str
            ).encode("utf-8")
            st.success("Download ready")

        if st.session_state.save_json_bytes:
            st.download_button(
                "Download relationships.json",
                data=st.session_state.save_json_bytes,
                file_name=save_name,
                mime="application/json",
                use_container_width=True,
            )

        upload_file = st.file_uploader(
            "Load relationships.json from your computer",
            type=["json"],
            key="relationships_json_upload",
            help="Use Browse to select a local relationships JSON file.",
        )
        if upload_file is not None and st.button(
            "Load uploaded JSON", use_container_width=True
        ):
            try:
                uploaded_data = json.load(upload_file)
                st.session_state.relationship_graph = RelationshipGraph()
                st.session_state.relationship_graph.from_dict(
                    uploaded_data.get("graph", {})
                )
                st.success("Loaded uploaded relationships JSON")
            except Exception as e:
                st.error(f"Failed to load uploaded JSON: {e}")

    # ==============================
    # Export
    # ==============================
    with st.expander("Export", expanded=False):
        relationship_set_name = st.text_input("Relationship Set Name", value="My Relationships")

        include_ai = False
        if st.session_state.ai_analysis_run and st.session_state.ai_result:
            include_ai = st.checkbox(
                "Include AI column descriptions",
                value=False,
                key="export_include_ai",
                help="Add AI-generated descriptions to each column in the export.",
            )

        include_suggested = st.checkbox(
            "Include suggested relationships",
            value=False,
            key="export_include_suggested",
            help="Add suggested (unconfirmed) relationships and their evidence to the export.",
        )

        if st.button("Export Markdown", use_container_width=True):
            databases = [
                {"server": info["server"], "database": info["database"]}
                for info in st.session_state.connections.values()
            ]
            ai_descs = _build_ai_description_lookup()
            md = generate_markdown(
                st.session_state.relationship_graph,
                relationship_set_name,
                databases,
                include_ai_descriptions=include_ai,
                ai_descriptions=ai_descs if include_ai else None,
                include_suggested=include_suggested,
            )
            st.download_button(
                "Download .md",
                data=md,
                file_name=f"{relationship_set_name.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        if st.button("Export JSON", use_container_width=True):
            databases = [
                {"server": info["server"], "database": info["database"]}
                for info in st.session_state.connections.values()
            ]
            ai_descs = _build_ai_description_lookup()
            report = generate_json_report(
                st.session_state.relationship_graph,
                relationship_set_name,
                databases,
                include_ai_descriptions=include_ai,
                ai_descriptions=ai_descs if include_ai else None,
                include_suggested=include_suggested,
            )
            json_bytes = json.dumps(report, indent=2, default=str).encode("utf-8")
            st.download_button(
                "Download .json",
                data=json_bytes,
                file_name=f"{relationship_set_name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

    # ==============================
    # Drift Monitoring
    # ==============================
    with st.expander("Drift Monitoring", expanded=False):
        st.subheader(
            "Schema Drift Monitoring",
            help=(
                "Detects how your database schema changes over time.\n\n"
                "1. **Download snapshot** records the current tables, columns, types, "
                "and nullability of your loaded tables to a JSON file you save locally.\n"
                "2. **Check Drift** compares the tables currently loaded against a "
                "snapshot you upload and reports added/removed tables, added/removed "
                "columns, and changed data types or nullability.\n\n"
                "Use it to catch upstream schema changes that may invalidate previously "
                "confirmed relationships."
            ),
        )
        st.caption(
            "Download a snapshot of the current schema, then upload it later to "
            "compare and see what tables and columns have changed."
        )

        if st.button("Prepare snapshot download", use_container_width=True):
            if st.session_state.table_metadata:
                snapshot = build_snapshot(st.session_state.table_metadata)
                st.session_state.snapshot_bytes = json.dumps(
                    snapshot, indent=2, default=str
                ).encode("utf-8")
                st.success("Snapshot ready")
            else:
                st.error("No tables loaded to snapshot")

        if st.session_state.snapshot_bytes:
            st.download_button(
                "Download snapshot",
                data=st.session_state.snapshot_bytes,
                file_name="schema_snapshot.json",
                mime="application/json",
                use_container_width=True,
            )

        prev_snapshot_file = st.file_uploader(
            "Previous snapshot to compare against",
            type=["json"],
            key="drift_snapshot_upload",
            help="Upload a snapshot you downloaded earlier.",
        )
        if prev_snapshot_file is not None and st.button(
            "Check Drift", use_container_width=True
        ):
            if not st.session_state.table_metadata:
                st.error("No tables loaded to compare.")
            else:
                try:
                    prev = json.load(prev_snapshot_file)
                except Exception as e:
                    prev = None
                    st.error(f"Invalid snapshot file: {e}")
                if prev:
                    drift = detect_drift(st.session_state.table_metadata, prev)
                    st.json(drift.to_dict())
                    md_drift = drift.to_markdown()
                    st.download_button(
                        "Download drift report",
                        data=md_drift,
                        file_name="drift_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

    # ==============================
    # Config Feedback
    # ==============================
    with st.expander("Config Feedback", expanded=False):
        st.caption("Suggest new column aliases from your review decisions.")

        # Show current aliases
        current_aliases = st.session_state.config.get("aliases", {})
        if current_aliases:
            st.markdown("**Current alias groups:**")
            for group, names in current_aliases.items():
                st.caption(f"  `{group}`: {', '.join(f'`{n}`' for n in names)}")

        # Add new alias
        with st.form("alias_form", clear_on_submit=False):
            alias_group = st.text_input(
                "Alias group name", placeholder="e.g., wellbore"
            )
            alias_names = st.text_input(
                "Column names (comma-separated)",
                placeholder="e.g., wb_id, wellbore_id, wellbore_number",
            )
            alias_submitted = st.form_submit_button(
                "Add Alias Group", use_container_width=True
            )

        if alias_submitted and alias_group and alias_names:
            names_list = [n.strip() for n in alias_names.split(",") if n.strip()]
            if names_list:
                if "aliases" not in st.session_state.config:
                    st.session_state.config["aliases"] = {}
                st.session_state.config["aliases"][alias_group] = names_list
                st.success(
                    f"Added alias group '{alias_group}' with {len(names_list)} names"
                )
                st.rerun()

        # Export updated config
        if st.button("Export Updated Config", use_container_width=True):
            config_yaml = yaml.dump(st.session_state.config, default_flow_style=False)
            st.download_button(
                "Download config.yaml",
                data=config_yaml,
                file_name="updated_config.yaml",
                mime="text/yaml",
                use_container_width=True,
            )

    # ==============================
    # Saved Graphs (graph store)
    # ==============================
    _render_graph_store_expander()

    # ==============================
    # Annotations
    # ==============================
    with st.expander("Annotations", expanded=False):
        if st.session_state.relationship_graph.nodes:
            # Node annotations
            node_options = st.session_state.relationship_graph.nodes
            selected_node = st.selectbox(
                "Select table to annotate",
                options=node_options,
                format_func=lambda x: x.split(".")[-1],
            )
            if selected_node:
                current_text = st.session_state.relationship_graph.get_annotation(
                    selected_node
                )
                new_text = st.text_area(
                    "Table Description", value=current_text, height=80
                )
                if st.button("Save Annotation", use_container_width=True):
                    st.session_state.relationship_graph.set_annotation(
                        selected_node, new_text
                    )
                    st.success("Annotation saved")

            # Edge annotations
            if st.session_state.relationship_graph.confirmed_edges:
                st.markdown("---")
                st.caption("Edge Annotations")
                edge_options = [
                    f"{e.source_key.split('.')[-1]}.{e.source_column} -> {e.target_key.split('.')[-1]}.{e.target_column}"
                    for e in st.session_state.relationship_graph.confirmed_edges
                ]
                selected_edge_idx = st.selectbox(
                    "Select relationship to annotate",
                    options=list(range(len(edge_options))),
                    format_func=lambda i: edge_options[i],
                )
                if selected_edge_idx is not None:
                    edge = st.session_state.relationship_graph.confirmed_edges[
                        selected_edge_idx
                    ]
                    edge_key = f"{edge.source_key}->{edge.target_key}"
                    current_edge_text = st.session_state.relationship_graph.get_annotation(
                        edge_key
                    )
                    new_edge_text = st.text_area(
                        "Relationship Description",
                        value=current_edge_text,
                        height=80,
                        key="edge_annotation_text",
                    )
                    if st.button("Save Edge Annotation", use_container_width=True):
                        st.session_state.relationship_graph.set_annotation(
                            edge_key, new_edge_text
                        )
                        edge.annotation = new_edge_text
                        st.success("Edge annotation saved")
        else:
            st.caption("Load tables to annotate")


# ====================================================================
# RIGHT PANEL
# ====================================================================
def _render_graph_tab() -> None:
    """Render the Graph tab (extracted from the original right-panel body)."""
    # ---- Graph ----
    st.subheader("Graph")

    if st.session_state.relationship_graph.nodes:
        # Graph display controls
        _cfg_graph = st.session_state.get("config", {}).get("graph", {})
        _cfg_thresholds = st.session_state.get("config", {}).get("thresholds", {})
        _default_max = int(_cfg_graph.get("max_suggested_edges", 100))
        _band_floor = {
            "All bands":    0.0,
            "Medium+":      float(_cfg_thresholds.get("confidence_medium", 0.70)),
            "High only":    float(_cfg_thresholds.get("confidence_high",   0.85)),
            "Confirmed only": 1.1,  # above any real confidence score
        }

        _gcols = st.columns([2, 2])
        with _gcols[0]:
            _band_label = st.selectbox(
                "Show suggested",
                options=list(_band_floor.keys()),
                index=0,
                key="graph_band_filter",
                help="Minimum confidence band for suggested edges shown in the graph.",
            )
        with _gcols[1]:
            _max_edges = st.slider(
                "Max suggested edges",
                min_value=0,
                max_value=500,
                value=_default_max,
                step=25,
                key="graph_max_edges",
                help="Top-N suggested edges by confidence. 0 = hide all suggested edges.",
            )

        try:
            net, _total, _visible = st.session_state.relationship_graph.to_pyvis(
                max_suggested=_max_edges,
                min_confidence=_band_floor[_band_label],
            )
            html = net.generate_html()
            html_b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
            st.iframe(f"data:text/html;base64,{html_b64}", height=600)
            if _total > 0:
                st.caption(
                    f"Showing {_visible} of {_total} suggested connections "
                    f"(top {_max_edges} ≥ {_band_label})."
                )
        except Exception as e:
            st.error(f"Graph rendering error: {e}")
    else:
        st.info("Select and load tables from the left panel to build the graph.")

    # ---- AI Column Descriptions ----
    if st.session_state.ai_analysis_run and st.session_state.ai_result:
        ai_res = st.session_state.ai_result
        if ai_res.column_descriptions:
            with st.expander(
                f"AI Column Descriptions ({len(ai_res.column_descriptions)})",
                expanded=False,
            ):
                desc_rows = []
                for desc in ai_res.column_descriptions:
                    terms = ", ".join(desc.domain_terms) if desc.domain_terms else ""
                    desc_rows.append(
                        {
                            "Table": desc.table,
                            "Column": desc.column,
                            "Type": desc.data_type,
                            "Description": desc.description,
                            "Domain Terms": terms,
                        }
                    )
                desc_df = pd.DataFrame(desc_rows)
                st.dataframe(
                    desc_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Description": st.column_config.TextColumn(
                            "Description", width="medium"
                        ),
                        "Domain Terms": st.column_config.TextColumn(
                            "Domain Terms", width="small"
                        ),
                    },
                )

    # ---- Suggested Edges ----
    if st.session_state.relationship_graph.suggested_edges:
        with st.expander(
            f"Suggested Relationships ({len(st.session_state.relationship_graph.suggested_edges)} total)",
            expanded=False,
        ):
            def _edge_key(e) -> str:
                return f"{e.source_key}|{e.source_column}|{e.target_key}|{e.target_column}"

            def _apply_decision(edge, decision: str) -> None:
                if decision == "confirm":
                    st.session_state.relationship_graph.confirm_edge(
                        edge.source_key,
                        edge.source_column,
                        edge.target_key,
                        edge.target_column,
                        rel_type="one-to-many",
                    )
                else:
                    st.session_state.relationship_graph.dismiss_suggestion(
                        edge.source_key,
                        edge.source_column,
                        edge.target_key,
                        edge.target_column,
                    )

            # --- Filter controls ---
            _thresholds = st.session_state.config.get("thresholds", {})
            _conf_high = float(_thresholds.get("confidence_high", 0.85))
            _conf_med  = float(_thresholds.get("confidence_medium", 0.70))

            has_ai = any(
                e.evidence.get("source") == "ai"
                for e in st.session_state.relationship_graph.suggested_edges
            )

            def _reset_page() -> None:
                st.session_state.suggested_page = 1

            _fcols = st.columns([2, 2, 3])
            with _fcols[0]:
                _band_opts = ["All", "High", "Medium", "Low"]
                _band_choice = st.selectbox(
                    "Confidence",
                    options=_band_opts,
                    index=_band_opts.index(st.session_state.suggested_band_filter)
                    if st.session_state.suggested_band_filter in _band_opts else 0,
                    key="suggested_band_select",
                    on_change=_reset_page,
                )
                st.session_state.suggested_band_filter = _band_choice

            with _fcols[1]:
                _src_opts = ["All", "AI-discovered", "Pipeline"]
                if has_ai:
                    _src_choice = st.selectbox(
                        "Source",
                        options=_src_opts,
                        index=_src_opts.index(st.session_state.suggested_source_filter)
                        if st.session_state.suggested_source_filter in _src_opts else 0,
                        key="suggested_source_select",
                        on_change=_reset_page,
                    )
                    st.session_state.suggested_source_filter = _src_choice
                else:
                    _src_choice = "All"

            with _fcols[2]:
                _table_input = st.text_input(
                    "Table search",
                    value=st.session_state.suggested_table_filter,
                    placeholder="e.g. WELL or LEASE",
                    key="suggested_table_input",
                    on_change=_reset_page,
                )
                st.session_state.suggested_table_filter = _table_input

            # Apply filters
            _tbl_lower = _table_input.strip().lower()
            edges_to_show = []
            for e in st.session_state.relationship_graph.suggested_edges:
                _is_ai = e.evidence.get("source") == "ai"

                if _src_choice == "AI-discovered" and not _is_ai:
                    continue
                if _src_choice == "Pipeline" and _is_ai:
                    continue

                if _band_choice == "High" and e.confidence < _conf_high:
                    continue
                if _band_choice == "Medium" and not (_conf_med <= e.confidence < _conf_high):
                    continue
                if _band_choice == "Low" and e.confidence >= _conf_med:
                    continue

                if _tbl_lower and _tbl_lower not in e.source_key.lower() and _tbl_lower not in e.target_key.lower():
                    continue

                edges_to_show.append(e)

            # Prune stale selections (edges that no longer exist)
            visible_keys = {_edge_key(e) for e in edges_to_show}
            st.session_state.suggested_selected &= visible_keys

            # Pagination controls
            page_size_options = [10, 25, 50, 100]
            ctrl_cols = st.columns([1, 1, 2])
            with ctrl_cols[0]:
                new_page_size = st.selectbox(
                    "Per page",
                    options=page_size_options,
                    index=page_size_options.index(st.session_state.suggested_page_size)
                    if st.session_state.suggested_page_size in page_size_options
                    else 1,
                    key="suggested_page_size_select",
                )
                if new_page_size != st.session_state.suggested_page_size:
                    st.session_state.suggested_page_size = new_page_size
                    st.session_state.suggested_page = 1

            page_size = st.session_state.suggested_page_size
            total = len(edges_to_show)
            total_pages = max(1, (total + page_size - 1) // page_size)
            if st.session_state.suggested_page > total_pages:
                st.session_state.suggested_page = total_pages
            if st.session_state.suggested_page < 1:
                st.session_state.suggested_page = 1
            page = st.session_state.suggested_page
            start = (page - 1) * page_size
            end = start + page_size
            page_edges = edges_to_show[start:end]
            page_keys = [_edge_key(e) for e in page_edges]

            with ctrl_cols[1]:
                st.caption(f"Page {page} of {total_pages}")
                st.caption(f"{total} total")
            with ctrl_cols[2]:
                nav_cols = st.columns(2)
                with nav_cols[0]:
                    if st.button(
                        "Prev",
                        key="suggested_prev",
                        use_container_width=True,
                        disabled=page <= 1,
                    ):
                        st.session_state.suggested_page = max(1, page - 1)
                        st.rerun()
                with nav_cols[1]:
                    if st.button(
                        "Next",
                        key="suggested_next",
                        use_container_width=True,
                        disabled=page >= total_pages,
                    ):
                        st.session_state.suggested_page = min(total_pages, page + 1)
                        st.rerun()

            # Bulk selection + action row
            page_selected_count = sum(
                1 for k in page_keys if k in st.session_state.suggested_selected
            )
            all_on_page = page_selected_count == len(page_keys) and len(page_keys) > 0
            bulk_cols = st.columns([2, 1, 1, 1])
            with bulk_cols[0]:
                select_all = st.checkbox(
                    f"Select all on page ({len(page_keys)})",
                    value=all_on_page,
                    key=f"suggested_select_all_page_{page}",
                )
                if select_all and not all_on_page:
                    st.session_state.suggested_selected.update(page_keys)
                    st.rerun()
                elif not select_all and all_on_page:
                    for k in page_keys:
                        st.session_state.suggested_selected.discard(k)
                    st.rerun()

            selected_count = len(st.session_state.suggested_selected)
            with bulk_cols[1]:
                if st.button(
                    f"Approve ({selected_count})",
                    key="bulk_approve",
                    disabled=selected_count == 0,
                    use_container_width=True,
                ):
                    selected_edges = [
                        e
                        for e in st.session_state.relationship_graph.suggested_edges
                        if _edge_key(e) in st.session_state.suggested_selected
                    ]
                    for edge in selected_edges:
                        _apply_decision(edge, "confirm")
                    st.session_state.suggested_selected.clear()
                    st.rerun()
            with bulk_cols[2]:
                if st.button(
                    f"Deny ({selected_count})",
                    key="bulk_deny",
                    disabled=selected_count == 0,
                    use_container_width=True,
                ):
                    selected_edges = [
                        e
                        for e in st.session_state.relationship_graph.suggested_edges
                        if _edge_key(e) in st.session_state.suggested_selected
                    ]
                    for edge in selected_edges:
                        _apply_decision(edge, "reject")
                    st.session_state.suggested_selected.clear()
                    st.rerun()
            with bulk_cols[3]:
                if st.button(
                    "Clear",
                    key="bulk_clear",
                    disabled=selected_count == 0,
                    use_container_width=True,
                    help="Clear selection",
                ):
                    st.session_state.suggested_selected.clear()
                    st.rerun()

            st.divider()

            for edge in page_edges:
                ekey = _edge_key(edge)
                src_label = edge.source_key.split(".")[-1]
                tgt_label = edge.target_key.split(".")[-1]
                is_ai = edge.evidence.get("source") == "ai"
                source_badge = "AI" if is_ai else "Pipeline"
                reasoning = edge.evidence.get("reasoning", "")

                cols = st.columns([0.4, 3.6, 1, 1])
                with cols[0]:
                    checked = st.checkbox(
                        " ",
                        value=ekey in st.session_state.suggested_selected,
                        key=f"sel_{ekey}",
                        label_visibility="collapsed",
                    )
                    if checked:
                        st.session_state.suggested_selected.add(ekey)
                    else:
                        st.session_state.suggested_selected.discard(ekey)
                with cols[1]:
                    st.caption(
                        f"{src_label}.{edge.source_column} -> {tgt_label}.{edge.target_column}"
                    )
                    st.caption(
                        f"Confidence: {edge.confidence:.2f} ({edge.confidence_band}) [{source_badge}]"
                    )
                    if reasoning:
                        with st.expander("Reasoning"):
                            st.caption(reasoning)
                with cols[2]:
                    if st.button("Accept", key=f"acc_{ekey}"):
                        _apply_decision(edge, "confirm")
                        st.session_state.suggested_selected.discard(ekey)
                        st.rerun()
                with cols[3]:
                    if st.button("Dismiss", key=f"dis_{ekey}"):
                        _apply_decision(edge, "reject")
                        st.session_state.suggested_selected.discard(ekey)
                        st.rerun()

    # ---- Confirmed Edges Table ----
    st.subheader("Relationships")

    if st.session_state.relationship_graph.confirmed_edges:
        # Show each confirmed edge with a remove button
        for i, edge in enumerate(st.session_state.relationship_graph.confirmed_edges):
            src_label = edge.source_key.split(".")[-1]
            tgt_label = edge.target_key.split(".")[-1]
            cols = st.columns([4, 1])
            with cols[0]:
                st.caption(
                    f"{src_label}.{edge.source_column} -> "
                    f"{tgt_label}.{edge.target_column} "
                    f"({edge.rel_type})"
                )
            with cols[1]:
                if st.button(
                    "Remove",
                    key=f"rm_{i}_{edge.source_key}_{edge.target_key}",
                    help="Remove this relationship",
                ):
                    st.session_state.relationship_graph.remove_edge(
                        edge.source_key, edge.target_key
                    )
                    st.rerun()

        # Editable table for type/annotation changes
        edges_data = []
        for edge in st.session_state.relationship_graph.confirmed_edges:
            edges_data.append(
                {
                    "Source": edge.source_key.split(".")[-1],
                    "Source Column": edge.source_column,
                    "Target": edge.target_key.split(".")[-1],
                    "Target Column": edge.target_column,
                    "Type": edge.rel_type,
                    "Annotation": edge.annotation,
                }
            )

        df = pd.DataFrame(edges_data)
        edited_df = st.data_editor(
            df,
            column_config={
                "Source": st.column_config.TextColumn(
                    "Source", disabled=True, width="small"
                ),
                "Source Column": st.column_config.TextColumn(
                    "S Col", disabled=True, width="small"
                ),
                "Target": st.column_config.TextColumn(
                    "Target", disabled=True, width="small"
                ),
                "Target Column": st.column_config.TextColumn(
                    "T Col", disabled=True, width="small"
                ),
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=[
                        "one-to-one",
                        "one-to-many",
                        "many-to-one",
                        "many-to-many",
                    ],
                    width="small",
                ),
                "Annotation": st.column_config.TextColumn("Annotation", width="medium"),
            },
            num_rows="fixed",
            use_container_width=True,
            key="relationship_editor",
        )

        # Sync edits back to graph
        if edited_df is not None and not edited_df.equals(df):
            _sync_edges_from_df(edited_df)
    else:
        st.info(
            "No confirmed relationships. Accept suggested edges or use the dialog below to add manually."
        )

    # ---- Manual Add Relationship ----
    with st.expander("Add Relationship Manually"):
        if st.session_state.relationship_graph.nodes:
            node_list = st.session_state.relationship_graph.nodes
            source_key = st.selectbox(
                "Source Table",
                node_list,
                format_func=lambda x: x.split(".")[-1],
                key="add_src",
            )
            target_key = st.selectbox(
                "Target Table",
                node_list,
                format_func=lambda x: x.split(".")[-1],
                key="add_tgt",
            )

            # Get columns for selected tables
            src_node = st.session_state.relationship_graph.get_node(source_key)
            tgt_node = st.session_state.relationship_graph.get_node(target_key)
            src_cols = list(src_node.get("columns", {}).keys()) if src_node else []
            tgt_cols = list(tgt_node.get("columns", {}).keys()) if tgt_node else []

            source_col = st.selectbox(
                "Source Column", src_cols if src_cols else ["(no columns)"]
            )
            target_col = st.selectbox(
                "Target Column", tgt_cols if tgt_cols else ["(no columns)"]
            )

            rel_type = st.selectbox(
                "Relationship Type",
                ["one-to-many", "one-to-one", "many-to-one", "many-to-many"],
            )

            if st.button("Add Relationship", use_container_width=True, type="primary"):
                st.session_state.relationship_graph.confirm_edge(
                    source_key,
                    source_col,
                    target_key,
                    target_col,
                    rel_type=rel_type,
                )
                st.rerun()
        else:
            st.caption("Load tables first")


with right_col:
    _tab_graph, _tab_chat = st.tabs(["Graph", "Chat"])

    with _tab_graph:
        _render_graph_tab()

    with _tab_chat:
        _render_chat_tab()
