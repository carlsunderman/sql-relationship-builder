"""Schema drift monitoring.

Compares current metadata/profile snapshots against a previous run
to detect new tables, removed tables, column changes, and type changes.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.models import ColumnInfo, TableMetadata


@dataclass
class ColumnChange:
    """Describes a change to a single column."""
    column_name: str
    change_type: str  # "added", "removed", "type_changed", "nullability_changed"
    old_value: Any = None
    new_value: Any = None


@dataclass
class TableChange:
    """Describes changes to a single table."""
    table_key: str
    change_type: str  # "added", "removed", "columns_changed", "row_count_changed"
    column_changes: List[ColumnChange] = field(default_factory=list)
    old_row_count: int = 0
    new_row_count: int = 0


@dataclass
class DriftReport:
    """Complete drift report comparing two snapshots."""
    new_tables: List[str] = field(default_factory=list)
    removed_tables: List[str] = field(default_factory=list)
    changed_tables: List[TableChange] = field(default_factory=list)
    unchanged_tables: List[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.new_tables or self.removed_tables or self.changed_tables)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "new_tables": len(self.new_tables),
            "removed_tables": len(self.removed_tables),
            "changed_tables": len(self.changed_tables),
            "unchanged_tables": len(self.unchanged_tables),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "summary": self.summary,
            "has_drift": self.has_drift,
            "new_tables": self.new_tables,
            "removed_tables": self.removed_tables,
            "changed_tables": [
                {
                    "table_key": tc.table_key,
                    "change_type": tc.change_type,
                    "column_changes": [
                        {
                            "column_name": cc.column_name,
                            "change_type": cc.change_type,
                            "old_value": cc.old_value,
                            "new_value": cc.new_value,
                        }
                        for cc in tc.column_changes
                    ],
                    "old_row_count": tc.old_row_count,
                    "new_row_count": tc.new_row_count,
                }
                for tc in self.changed_tables
            ],
            "unchanged_tables": self.unchanged_tables,
        }

    def to_markdown(self) -> str:
        """Generate a markdown summary of drift."""
        lines: List[str] = []
        lines.append("# Schema Drift Report")
        lines.append("")
        lines.append(f"- **New tables:** {len(self.new_tables)}")
        lines.append(f"- **Removed tables:** {len(self.removed_tables)}")
        lines.append(f"- **Changed tables:** {len(self.changed_tables)}")
        lines.append(f"- **Unchanged tables:** {len(self.unchanged_tables)}")
        lines.append("")

        if self.new_tables:
            lines.append("## New Tables")
            lines.append("")
            for t in self.new_tables:
                lines.append(f"- `{t}`")
            lines.append("")

        if self.removed_tables:
            lines.append("## Removed Tables")
            lines.append("")
            for t in self.removed_tables:
                lines.append(f"- `{t}`")
            lines.append("")

        if self.changed_tables:
            lines.append("## Changed Tables")
            lines.append("")
            for tc in self.changed_tables:
                lines.append(f"### `{tc.table_key}`")
                lines.append("")
                if tc.old_row_count and tc.new_row_count:
                    lines.append(f"- **Row count:** {tc.old_row_count:,} -> {tc.new_row_count:,}")
                for cc in tc.column_changes:
                    if cc.change_type == "type_changed":
                        lines.append(f"- **Type change:** `{cc.column_name}`: `{cc.old_value}` -> `{cc.new_value}`")
                    elif cc.change_type == "nullability_changed":
                        lines.append(f"- **Nullability:** `{cc.column_name}`: {cc.old_value} -> {cc.new_value}")
                    elif cc.change_type == "added":
                        lines.append(f"- **Added column:** `{cc.column_name}` ({cc.new_value})")
                    elif cc.change_type == "removed":
                        lines.append(f"- **Removed column:** `{cc.column_name}` ({cc.old_value})")
                lines.append("")

        if not self.has_drift:
            lines.append("## No Drift Detected")
            lines.append("")
            lines.append("All tables and columns remain unchanged since the last snapshot.")
            lines.append("")

        return "\n".join(lines)


def _table_to_snapshot(meta: TableMetadata) -> Dict[str, Any]:
    """Convert a TableMetadata to a serializable snapshot dict.

    Args:
        meta: Table metadata.

    Returns:
        Serializable dictionary.
    """
    return {
        "schema_name": meta.schema_name,
        "table_name": meta.table_name,
        "row_count": meta.row_count,
        "columns": {
            c.name: {
                "data_type": c.data_type,
                "is_nullable": c.is_nullable,
                "ordinal_position": c.ordinal_position,
            }
            for c in meta.columns
        },
    }


def build_snapshot(inventory: Dict[str, TableMetadata]) -> Dict[str, Any]:
    """Build a serializable metadata snapshot in memory.

    Args:
        inventory: Current metadata inventory.

    Returns:
        Snapshot dictionary suitable for JSON serialization and later
        comparison via detect_drift.
    """
    return {
        "tables": {
            key: _table_to_snapshot(meta)
            for key, meta in inventory.items()
        },
    }


def save_snapshot(
    inventory: Dict[str, TableMetadata],
    filepath: str,
) -> None:
    """Save a metadata snapshot to disk for future drift comparison.

    Args:
        inventory: Current metadata inventory.
        filepath: Path to save the snapshot JSON.
    """
    snapshot = build_snapshot(inventory)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def load_snapshot(filepath: str) -> Dict[str, Any]:
    """Load a previously saved metadata snapshot.

    Args:
        filepath: Path to the snapshot JSON.

    Returns:
        Snapshot dictionary, or empty dict if file not found.
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_drift(
    current_inventory: Dict[str, TableMetadata],
    previous_snapshot: Dict[str, Any],
) -> DriftReport:
    """Compare current metadata against a previous snapshot.

    Detects:
    - New tables (present now, not in snapshot)
    - Removed tables (in snapshot, not present now)
    - Column additions/removals
    - Type changes
    - Nullability changes
    - Row count changes

    Args:
        current_inventory: Current metadata from build_inventory().
        previous_snapshot: Previously saved snapshot from save_snapshot().

    Returns:
        DriftReport with all detected changes.
    """
    report = DriftReport()
    prev_tables: Dict[str, Dict[str, Any]] = previous_snapshot.get("tables", {})

    current_keys: Set[str] = set(current_inventory.keys())
    prev_keys: Set[str] = set(prev_tables.keys())

    # New tables
    report.new_tables = sorted(current_keys - prev_keys)

    # Removed tables
    for key in sorted(prev_keys - current_keys):
        report.removed_tables.append(key)
        report.changed_tables.append(TableChange(
            table_key=key,
            change_type="removed",
        ))

    # Check existing tables for changes
    for key in sorted(current_keys & prev_keys):
        current_meta = current_inventory[key]
        prev_data = prev_tables[key]
        prev_cols: Dict[str, Dict[str, Any]] = prev_data.get("columns", {})
        curr_cols: Dict[str, Dict[str, Any]] = {
            c.name: {
                "data_type": c.data_type,
                "is_nullable": c.is_nullable,
            }
            for c in current_meta.columns
        }

        column_changes: List[ColumnChange] = []

        # Added columns
        for col_name in sorted(set(curr_cols.keys()) - set(prev_cols.keys())):
            column_changes.append(ColumnChange(
                column_name=col_name,
                change_type="added",
                new_value=curr_cols[col_name]["data_type"],
            ))

        # Removed columns
        for col_name in sorted(set(prev_cols.keys()) - set(curr_cols.keys())):
            column_changes.append(ColumnChange(
                column_name=col_name,
                change_type="removed",
                old_value=prev_cols[col_name]["data_type"],
            ))

        # Type changes
        for col_name in sorted(set(curr_cols.keys()) & set(prev_cols.keys())):
            if curr_cols[col_name]["data_type"] != prev_cols[col_name]["data_type"]:
                column_changes.append(ColumnChange(
                    column_name=col_name,
                    change_type="type_changed",
                    old_value=prev_cols[col_name]["data_type"],
                    new_value=curr_cols[col_name]["data_type"],
                ))
            if curr_cols[col_name]["is_nullable"] != prev_cols[col_name]["is_nullable"]:
                column_changes.append(ColumnChange(
                    column_name=col_name,
                    change_type="nullability_changed",
                    old_value=str(prev_cols[col_name]["is_nullable"]),
                    new_value=str(curr_cols[col_name]["is_nullable"]),
                ))

        # Row count change
        prev_row_count = prev_data.get("row_count", 0)
        row_count_changed = current_meta.row_count != prev_row_count

        if column_changes:
            report.changed_tables.append(TableChange(
                table_key=key,
                change_type="columns_changed",
                column_changes=column_changes,
                old_row_count=prev_row_count,
                new_row_count=current_meta.row_count,
            ))
        elif row_count_changed:
            report.changed_tables.append(TableChange(
                table_key=key,
                change_type="row_count_changed",
                old_row_count=prev_row_count,
                new_row_count=current_meta.row_count,
            ))
        else:
            report.unchanged_tables.append(key)

    return report
