"""JSON save/load for relationship persistence.

Serializes and deserializes the complete relationship state including
connection metadata, selected tables, graph state, and annotations.
Credentials are excluded from saved state for security.
"""

import json
import os
from typing import Any, Dict, List, Optional


def save_relationships(filepath: str, data: Dict[str, Any]) -> None:
    """Save relationship state to a JSON file.

    Args:
        filepath: Path to the output JSON file.
        data: Serializable relationship state dictionary.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_relationships(filepath: str) -> Dict[str, Any]:
    """Load relationship state from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Loaded state dictionary, or empty dict if file not found.
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_save_data(
    connections: List[Dict[str, str]],
    selected_tables: List[str],
    graph_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a serializable save payload from UI state.

    Args:
        connections: List of connection metadata dicts (server, database only -- no credentials).
        selected_tables: List of selected table keys.
        graph_dict: Dictionary from RelationshipGraph.to_dict().

    Returns:
        Complete save payload.
    """
    return {
        "version": "0.1.0",
        "connections": connections,
        "selected_tables": selected_tables,
        "graph": graph_dict,
    }


def extract_connections_from_save(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract connection metadata from a loaded save payload.

    Args:
        data: Loaded save payload.

    Returns:
        List of connection metadata dicts.
    """
    return data.get("connections", [])


def extract_selected_tables(data: Dict[str, Any]) -> List[str]:
    """Extract selected table keys from a loaded save payload.

    Args:
        data: Loaded save payload.

    Returns:
        List of table keys.
    """
    return data.get("selected_tables", [])


def extract_graph_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract graph state from a loaded save payload.

    Args:
        data: Loaded save payload.

    Returns:
        Graph state dictionary for RelationshipGraph.from_dict().
    """
    return data.get("graph", {})