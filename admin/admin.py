"""Streamlit admin panel for managing AI provider configurations.

Run with:
    streamlit run admin/admin.py --server.port 8550
"""

import json
import os
import secrets
import uuid
from typing import Any, Dict, Optional

import streamlit as st

# Load config_store via absolute path (works when admin/admin.py is run directly).
import importlib.util as _importlib_util

_config_store_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_store.py"
)
_spec = _importlib_util.spec_from_file_location("config_store", _config_store_path)
_config_store_module = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_config_store_module)

# Re-export all public names from config_store at module level.
ALL_PROVIDERS = _config_store_module.ALL_PROVIDERS
ProviderConfig = _config_store_module.ProviderConfig
ServerConnection = _config_store_module.ServerConnection
ADMIN_DB_PATH = _config_store_module.ADMIN_DB_PATH
encrypt_value = _config_store_module.encrypt_value
decrypt_value = _config_store_module.decrypt_value
has_admin_password = _config_store_module.has_admin_password
init_db = _config_store_module.init_db
list_configs = _config_store_module.list_configs
save_config = _config_store_module.save_config
delete_config = _config_store_module.delete_config
get_config = _config_store_module.get_config
get_active_config = _config_store_module.get_active_config
set_admin_password = _config_store_module.set_admin_password
verify_admin_password = _config_store_module.verify_admin_password
seed_from_env = _config_store_module.seed_from_env
to_llm_config = _config_store_module.to_llm_config
list_server_connections = _config_store_module.list_server_connections
get_server_connection = _config_store_module.get_server_connection
save_server_connection = _config_store_module.save_server_connection
delete_server_connection = _config_store_module.delete_server_connection
get_connection_password = _config_store_module.get_connection_password
get_allowed_databases = _config_store_module.get_allowed_databases

# Load src.graph_store via path injection (admin/admin.py runs from project root).
import sys as _sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
import src.graph_store as _graph_store  # noqa: E402

_graph_store.init_db()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Config Admin", layout="wide")

init_db()
seed_from_env()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _setup_initial_password() -> None:
    """One-time password setup on first run."""
    st.title("Initial Setup")
    st.warning(
        "No admin password is set. Create one below. "
        "You can also change it later in Settings."
    )

    with st.form("setup_form"):
        password = st.text_input("Admin Password", type="password", key="setup_pw")
        confirm = st.text_input("Confirm Password", type="password", key="setup_confirm")
        submitted = st.form_submit_button("Set Password", key="submit_setup_password")

    if submitted:
        if not password:
            st.error("Password cannot be empty.")
        elif password != confirm:
            st.error("Passwords do not match.")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters.")
        else:
            set_admin_password(password)
            st.session_state.logged_in = True
            return


def _show_login() -> None:
    st.title("AI Config Admin")

    with st.form("login_form"):
        password = st.text_input("Password", type="password", key="login_pw")
        submitted = st.form_submit_button("Login", key="submit_login")

    if submitted:
        if verify_admin_password(password):
            st.session_state.logged_in = True
            return
        else:
            st.error("Incorrect password.")


def _logout() -> None:
    st.session_state.logged_in = False



# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

def _show_admin() -> None:
    st.title("AI Config Admin")

    col_left, col_right = st.columns([0.7, 0.3])
    with col_right:
        if st.button("Logout", use_container_width=True, key="logout"):
            _logout()

    tab_list, tab_edit, tab_connections, tab_graphs, tab_settings = st.tabs([
        "Configs", "Edit Config", "Server Connections", "Saved Graphs", "Settings",
    ])

    with tab_list:
        _show_config_list()

    with tab_edit:
        _show_config_editor()

    with tab_connections:
        _show_server_connections()

    with tab_graphs:
        _show_saved_graphs()

    with tab_settings:
        _show_settings()


# ---------------------------------------------------------------------------
# Config list
# ---------------------------------------------------------------------------

def _show_config_list() -> None:
    st.header("Provider Configs")

    configs = list_configs()

    if not configs:
        st.info("No configs yet. Go to the 'Edit Config' tab to create one.")
        return

    for cfg in configs:
        active_badge = "● Active" if cfg.is_active else ""
        expanded = st.expander(
            f"**{cfg.name}** ({cfg.provider}) {active_badge}",
            expanded=cfg.is_active,
        )
        with expanded:
            masked = cfg.to_dict(mask_api_key=True)
            st.json(masked, expanded=False)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Set Active", key=f"active_{cfg.id}", use_container_width=True
                ):
                    cfg.is_active = True
                    save_config(cfg)
            with col2:
                if st.button(
                    "Delete", key=f"del_{cfg.id}", use_container_width=True
                ):
                    delete_config(cfg.id)


# ---------------------------------------------------------------------------
# Config editor
# ---------------------------------------------------------------------------

def _show_config_editor() -> None:
    existing = list_configs()
    options = ["<New config>"] + [f"{c.name} ({c.provider})" for c in existing]
    label_to_id = {
        f"{c.name} ({c.provider})": c.id for c in existing
    }

    picked = st.selectbox(
        "Select config to edit",
        options,
        key="config_editor_pick",
    )

    editing_id = label_to_id.get(picked)
    config: Optional[ProviderConfig] = None
    is_new = False

    if editing_id:
        config = get_config(editing_id)
        if config is None:
            st.error("Config not found.")
            return
    else:
        is_new = True
        config = ProviderConfig()

    st.header(f"{'Edit' if not is_new else 'New'} Config")

    with st.form("config_form", clear_on_submit=False):
        config.name = st.text_input(
            "Name", value=config.name,
            help="Human-readable label (e.g. 'Production Azure')",
        )
        config.provider = st.selectbox(
            "Provider",
            options=ALL_PROVIDERS,
            index=ALL_PROVIDERS.index(config.provider) if config.provider in ALL_PROVIDERS else 0,
        )

        config.endpoint = st.text_input(
            "Endpoint", value=config.endpoint,
            help="Base URL (Azure: resource endpoint, Local: Ollama/LM Studio URL)",
        )
        config.model = st.text_input(
            "Model", value=config.model,
            help="Model name (Azure: ignored, use Deployment Name)",
        )

        api_key = st.text_input(
            "API Key", type="password", value="", key="config_api_key",
            help="Leave blank to keep existing key unchanged.",
        )
        # Store whether user entered a new key.
        st.session_state._new_api_key = api_key

        config.api_version = st.text_input(
            "API Version", value=config.api_version,
            help="Azure: api-version, Anthropic: anthropic-version",
        )
        config.deployment_name = st.text_input(
            "Deployment Name", value=config.deployment_name,
            help="Azure only: deployment name in Azure OpenAI",
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            config.temperature = st.number_input(
                "Temperature", value=config.temperature, min_value=0.0, max_value=2.0, step=0.1,
            )
        with col_b:
            config.max_tokens = st.number_input(
                "Max Tokens", value=config.max_tokens, min_value=256, max_value=131072, step=256,
            )
        with col_c:
            config.timeout = st.number_input(
                "Timeout (s)", value=config.timeout, min_value=10, max_value=600, step=10,
            )

        config.verify_ssl = st.checkbox(
            "Verify SSL", value=config.verify_ssl,
        )

        # Anthropic-specific fields
        if config.provider == "anthropic":
            st.subheader("Anthropic-specific")
            extra = json.loads(config.extra_config or "{}")
            av = st.text_input(
                "Anthropic Version",
                value=extra.get("anthropic_version", "2023-06-01"),
            )
            extra["anthropic_version"] = av
            config.extra_config = json.dumps(extra)

        config.is_active = st.checkbox(
            "Set as active config", value=config.is_active,
            help="Only one config can be active at a time.",
        )

        submitted = st.form_submit_button(
            "Save Config", use_container_width=True, type="primary", key="submit_save_config"
        )

    if submitted:
        if not config.name.strip():
            st.error("Name is required.")
            return

        # Apply new API key if provided.
        new_key = st.session_state.get("_new_api_key", "")
        if new_key:
            config.api_key_encrypted = encrypt_value(new_key)
        elif is_new:
            # New config with no key provided — keep empty.
            pass

        if is_new:
            config.id = str(uuid.uuid4())

        save_config(config)
        st.success(f"Config '{config.name}' saved.")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _show_settings() -> None:
    st.header("Settings")

    with st.form("change_password_form"):
        st.subheader("Change Password")
        current_pw = st.text_input(
            "Current Password", type="password", key="change_current_pw"
        )
        new_pw = st.text_input(
            "New Password", type="password", key="change_new_pw"
        )
        confirm_pw = st.text_input(
            "Confirm New Password", type="password", key="change_confirm_pw"
        )
        pw_submitted = st.form_submit_button("Change Password", key="submit_change_password")

    if pw_submitted:
        if not verify_admin_password(current_pw):
            st.error("Current password is incorrect.")
        elif new_pw != confirm_pw:
            st.error("New passwords do not match.")
        elif len(new_pw) < 8:
            st.error("Password must be at least 8 characters.")
        else:
            set_admin_password(new_pw)
            st.success("Password changed.")

    # Encryption key info
    st.subheader("Encryption Key")
    key_env = os.environ.get("ADMIN_ENCRYPTION_KEY")
    if key_env:
        st.info("Using encryption key from ADMIN_ENCRYPTION_KEY env var.")
    else:
        st.info(
            "Using auto-generated key file. "
            "For production, set ADMIN_ENCRYPTION_KEY env var."
        )

    # DB info
    st.subheader("Database")
    st.write(f"Path: `{ADMIN_DB_PATH}`")
    st.write(f"Configs: {len(list_configs())}")


# ---------------------------------------------------------------------------
# Server Connections
# ---------------------------------------------------------------------------

def _show_server_connections() -> None:
    st.header("Server Connections")

    if st.button("New Connection", use_container_width=True, key="new_connection"):
        st.session_state.pop("editing_connection_id", None)
        st.session_state._show_new_connection = True
        st.rerun()

    connections = list_server_connections()

    if not connections:
        st.info("No server connections yet. Click 'New Connection' to create one.")
    else:
        for sc in connections:
            dbs = json.loads(sc.allowed_databases or "[]")
            db_label = f"{len(dbs)} database(s)" if dbs else "discover all"
            expanded = st.expander(f"**{sc.name}** ({sc.server}) — {db_label}")
            with expanded:
                st.caption(f"Username: {sc.username or '(none)'}")
                st.caption(f"Password: {'****' if sc.password_encrypted else '(none)'}")
                if dbs:
                    st.caption(f"Allowed databases: {', '.join(dbs)}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit", key=f"edit_sc_{sc.id}", use_container_width=True):
                        st.session_state.editing_connection_id = sc.id
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"del_sc_{sc.id}", use_container_width=True):
                        delete_server_connection(sc.id)
                        st.rerun()

    # Editor
    editing_id = st.session_state.get("editing_connection_id")
    if editing_id or st.session_state.get("_show_new_connection"):
        _show_connection_editor()


def _show_connection_editor() -> None:
    editing_id = st.session_state.get("editing_connection_id")
    sc: Optional[ServerConnection] = None
    is_new = False

    if editing_id:
        sc = get_server_connection(editing_id)
        if sc is None:
            st.error("Connection not found.")
            return
    else:
        is_new = True
        sc = ServerConnection()

    st.subheader(f"{'Edit' if not is_new else 'New'} Connection")

    with st.form("connection_form", clear_on_submit=False):
        sc.name = st.text_input(
            "Name", value=sc.name,
            help="Human-readable label (e.g. 'Production SQL Server')",
        )
        sc.server = st.text_input(
            "Server", value=sc.server,
            help="Server hostname or IP (e.g. 'localhost' or 'sql-prod.company.com\\SQLEXPRESS')",
        )
        sc.username = st.text_input(
            "Username", value=sc.username,
            help="SQL Server login name",
        )

        password = st.text_input(
            "Password", type="password", value="", key="conn_password",
            help="Leave blank to keep existing password unchanged.",
        )
        st.session_state._new_conn_password = password

        # Allowed databases: comma-separated or leave empty to discover all
        current_dbs = json.loads(sc.allowed_databases or "[]")
        dbs_input = st.text_input(
            "Allowed Databases",
            value=", ".join(current_dbs) if current_dbs else "",
            help="Comma-separated list of databases this connection can access. "
                 "Leave empty to discover all databases at connect time.",
        )

        submitted = st.form_submit_button(
            "Save Connection", use_container_width=True, type="primary", key="submit_save_connection"
        )

    if st.button("Cancel", use_container_width=True, key="cancel_connection"):
        st.session_state.pop("editing_connection_id", None)
        st.rerun()

    if submitted:
        if not sc.name.strip():
            st.error("Name is required.")
            return
        if not sc.server.strip():
            st.error("Server is required.")
            return

        new_pw = st.session_state.get("_new_conn_password", "")
        if new_pw:
            sc.password_encrypted = encrypt_value(new_pw)
        elif is_new:
            pass  # new connection with no password is allowed (Windows auth)

        if dbs_input:
            sc.allowed_databases = json.dumps(
                [d.strip() for d in dbs_input.split(",") if d.strip()]
            )
        else:
            sc.allowed_databases = "[]"

        if is_new:
            sc.id = str(uuid.uuid4())

        save_server_connection(sc)
        st.success(f"Connection '{sc.name}' saved.")
        st.session_state.pop("editing_connection_id", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Saved Graphs management
# ---------------------------------------------------------------------------

def _show_saved_graphs() -> None:
    """Render the Saved Graphs tab: list, edit, delete, fork."""
    st.header("Saved Graphs")

    graphs = _graph_store.list_graphs()

    if not graphs:
        st.info(
            "No saved graphs yet. Save a graph from the main app's "
            "'Saved Graphs' expander to populate this list."
        )
        return

    # Summary metrics
    total_nodes = sum(len(_graph_store.list_nodes(g.id)) for g in graphs)
    total_edges = sum(len(_graph_store.list_edges(g.id)) for g in graphs)
    template_count = sum(1 for g in graphs if g.is_template)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Graphs", len(graphs))
    metric_cols[1].metric("Templates", template_count)
    metric_cols[2].metric("Total Nodes", total_nodes)
    metric_cols[3].metric("Total Edges", total_edges)

    st.markdown("---")
    st.markdown("### Graphs")

    # Search/filter
    filter_cols = st.columns([3, 1])
    with filter_cols[0]:
        search = st.text_input(
            "Filter by name or description",
            placeholder="Search...",
            key="graph_admin_search",
            label_visibility="collapsed",
        )
    with filter_cols[1]:
        show_only = st.selectbox(
            "Type",
            options=["All", "Templates only", "Forks only"],
            key="graph_admin_filter_type",
            label_visibility="collapsed",
        )

    filtered = _filter_graphs(graphs, search, show_only)

    if not filtered:
        st.caption("No graphs match the current filter.")
    else:
        for g in filtered:
            _render_graph_admin_row(g)

    # Editor panel (shown when editing)
    if st.session_state.get("editing_graph_id"):
        _show_graph_editor(st.session_state.editing_graph_id)


def _filter_graphs(graphs, search: str, show_only: str):
    """Apply search text and type filter to the graph list."""
    result = graphs
    if show_only == "Templates only":
        result = [g for g in result if g.is_template]
    elif show_only == "Forks only":
        result = [g for g in result if not g.is_template]
    if search:
        needle = search.lower()
        result = [
            g for g in result
            if needle in g.name.lower()
            or needle in (g.description or "").lower()
            or needle in g.database_server.lower()
        ]
    return result


def _render_graph_admin_row(g) -> None:
    """Render a single graph row with view/edit/delete/fork buttons."""
    node_count = len(_graph_store.list_nodes(g.id))
    edge_count = len(_graph_store.list_edges(g.id))
    fork_label = (
        f" (fork of {_truncate(g.parent_graph_id, 8)})"
        if g.parent_graph_id
        else ""
    )
    template_badge = " [TEMPLATE]" if g.is_template else ""
    title = f"**{g.name}**{template_badge}{fork_label}"

    with st.expander(f"{title} — {node_count} tables, {edge_count} edges"):
        info_cols = st.columns(2)
        with info_cols[0]:
            st.caption(f"ID: `{g.id}`")
            st.caption(f"Server: `{g.database_server or '(none)'}`")
            if g.description:
                st.caption(f"Description: {g.description}")
        with info_cols[1]:
            st.caption(f"Created: {g._format_ts(g.created_at)}")
            st.caption(f"Updated: {g._format_ts(g.updated_at)}")
            st.caption(f"Version: {g.version}")
            st.caption(f"Domain tag: {g.domain_tag or '(none)'}")

        # Node/edge preview
        with st.expander(f"Preview contents ({node_count} nodes, {edge_count} edges)", expanded=False):
            nodes = _graph_store.list_nodes(g.id)
            if nodes:
                st.markdown("**Tables:**")
                for n in nodes[:20]:
                    st.caption(f"  - `{n.full_table_name}` (~{n.row_count} rows)")
                if len(nodes) > 20:
                    st.caption(f"  ... and {len(nodes) - 20} more")

            edges = _graph_store.list_edges(g.id)
            if edges:
                st.markdown("**Relationships:**")
                for e in edges[:20]:
                    src = e.source_table_key.split(".")[-1]
                    tgt = e.target_table_key.split(".")[-1]
                    st.caption(
                        f"  - `{src}.{e.source_column}` -> "
                        f"`{tgt}.{e.target_column}` ({e.edge_type})"
                    )
                if len(edges) > 20:
                    st.caption(f"  ... and {len(edges) - 20} more")

        # Action buttons
        action_cols = st.columns(4)
        with action_cols[0]:
            if st.button("Edit", key=f"edit_g_{g.id}", use_container_width=True):
                st.session_state.editing_graph_id = g.id
                st.rerun()
        with action_cols[1]:
            if st.button("Fork", key=f"fork_g_{g.id}", use_container_width=True):
                _admin_fork_graph(g)
        with action_cols[2]:
            export_data = json.dumps(_graph_store.export_graph(g.id), indent=2, default=str)
            st.download_button(
                "Export",
                data=export_data,
                file_name=f"{g.name.replace(' ', '_')}.json",
                mime="application/json",
                key=f"export_g_{g.id}",
                use_container_width=True,
            )
        with action_cols[3]:
            armed = bool(st.session_state.get(f"confirm_del_{g.id}"))
            if st.button(
                "Confirm delete?" if armed else "Delete",
                key=f"del_g_{g.id}",
                type="secondary" if not armed else "primary",
                use_container_width=True,
            ):
                _confirm_delete_graph(g)


def _confirm_delete_graph(g) -> None:
    """Two-step delete: first 'Delete' click arms, second click deletes.

    Only invoked from the Delete button handler, so a pending arm flag
    means this run is the confirming second click.
    """
    confirm_key = f"confirm_del_{g.id}"
    if st.session_state.get(confirm_key):
        _graph_store.delete_graph(g.id)
        # Clean up session state if we were editing this graph
        if st.session_state.get("editing_graph_id") == g.id:
            st.session_state.pop("editing_graph_id", None)
        st.session_state.pop(confirm_key, None)
        st.success(f"Deleted graph '{g.name}'")
        st.rerun()
    else:
        st.session_state[confirm_key] = True
        st.warning(
            f"Click 'Delete' again to confirm removing **{g.name}** "
            f"({len(_graph_store.list_nodes(g.id))} tables, "
            f"{len(_graph_store.list_edges(g.id))} edges). "
            "This cannot be undone."
        )


def _admin_fork_graph(g) -> None:
    """Create a fork of an existing graph from the admin panel."""
    new_name = f"{g.name} (fork)"
    try:
        fork = _graph_store.fork_graph(
            source_id=g.id,
            name=new_name,
            created_by="admin",
        )
        st.success(f"Forked as '{fork.name}'")
        st.rerun()
    except Exception as e:
        st.error(f"Fork failed: {e}")


def _show_graph_editor(graph_id: str) -> None:
    """Inline editor for a graph's metadata."""
    g = _graph_store.get_graph(graph_id)
    if g is None:
        st.error("Graph no longer exists.")
        st.session_state.pop("editing_graph_id", None)
        return

    st.markdown("---")
    st.markdown(f"### Edit Graph: `{g.name}`")

    with st.form(f"graph_editor_{graph_id}"):
        new_name = st.text_input("Name", value=g.name)
        new_description = st.text_area("Description", value=g.description or "", height=80)
        new_domain = st.text_input("Domain tag", value=g.domain_tag or "")
        new_db_server = st.text_input(
            "Database server (server\\database)",
            value=g.database_server or "",
            help="Used for auto-connect in chat. Use format: server\\database.",
        )
        new_is_template = st.checkbox(
            "Template (curated baseline, not editable by end users)",
            value=g.is_template,
        )
        save_cols = st.columns([1, 1, 4])
        with save_cols[0]:
            submitted = st.form_submit_button("Save", use_container_width=True, type="primary", key="submit_save_graph")
        with save_cols[1]:
            cancel = st.form_submit_button("Cancel", use_container_width=True, key="cancel_graph")

    if submitted:
        g.name = new_name.strip()
        g.description = new_description.strip()
        g.domain_tag = new_domain.strip()
        g.database_server = new_db_server.strip()
        g.is_template = new_is_template
        g.version += 1
        _graph_store.save_graph(g)
        # Save snapshot at this version
        snap = _graph_store.GraphSnapshot(
            graph_id=g.id,
            version=g.version,
            snapshot_json=json.dumps(_graph_store.export_graph(g.id), default=str),
        )
        _graph_store.save_snapshot(snap)
        st.session_state.pop("editing_graph_id", None)
        st.success(f"Saved '{g.name}' (v{g.version})")
        st.rerun()
    if cancel:
        st.session_state.pop("editing_graph_id", None)
        st.rerun()


def _truncate(text: str, n: int) -> str:
    """Return text truncated to n characters with ellipsis if needed."""
    if not text:
        return ""
    return text if len(text) <= n else text[:n] + "..."


# Extend StoredGraph with a helper to format timestamps for display.
def _format_ts(self, ts: str) -> str:
    if not ts:
        return "(unknown)"
    try:
        from datetime import datetime as _dt
        dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return ts

_graph_store.StoredGraph._format_ts = _format_ts



# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not has_admin_password():
    # First run — set initial password.
    _setup_initial_password()

if not st.session_state.logged_in:
    _show_login()
else:
    _show_admin()
