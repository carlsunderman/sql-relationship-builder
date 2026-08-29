"""Tests for schema drift monitoring."""

import json
import os
import tempfile

from src.drift import (
    ColumnChange,
    DriftReport,
    TableChange,
    detect_drift,
    load_snapshot,
    save_snapshot,
)
from src.models import ColumnInfo, TableMetadata


def _make_metadata(schema: str, table: str, columns: list, row_count: int = 100) -> TableMetadata:
    return TableMetadata(
        schema_name=schema,
        table_name=table,
        row_count=row_count,
        columns=[
            ColumnInfo(name=n, data_type=t, is_nullable=nullable, ordinal_position=i)
            for i, (n, t, nullable) in enumerate(columns)
        ],
    )


class TestDriftReport:
    def test_no_drift(self):
        report = DriftReport()
        assert not report.has_drift
        assert report.summary == {
            "new_tables": 0,
            "removed_tables": 0,
            "changed_tables": 0,
            "unchanged_tables": 0,
        }

    def test_has_drift_new_tables(self):
        report = DriftReport(new_tables=["dbo.new_table"])
        assert report.has_drift

    def test_has_drift_removed_tables(self):
        report = DriftReport(removed_tables=["dbo.old_table"])
        assert report.has_drift

    def test_has_drift_changed_tables(self):
        report = DriftReport(changed_tables=[
            TableChange(table_key="dbo.changes", change_type="columns_changed"),
        ])
        assert report.has_drift

    def test_to_dict(self):
        report = DriftReport(
            new_tables=["dbo.new"],
            removed_tables=["dbo.old"],
            changed_tables=[
                TableChange(
                    table_key="dbo.chg",
                    change_type="columns_changed",
                    column_changes=[
                        ColumnChange(column_name="x", change_type="added", new_value="int"),
                    ],
                    old_row_count=100,
                    new_row_count=150,
                ),
            ],
            unchanged_tables=["dbo.same"],
        )
        d = report.to_dict()
        assert d["has_drift"] is True
        assert d["summary"]["new_tables"] == 1
        assert d["summary"]["removed_tables"] == 1
        assert d["summary"]["changed_tables"] == 1
        assert d["summary"]["unchanged_tables"] == 1

    def test_to_markdown_empty(self):
        report = DriftReport()
        md = report.to_markdown()
        assert "# Schema Drift Report" in md
        assert "No Drift Detected" in md

    def test_to_markdown_with_changes(self):
        report = DriftReport(
            new_tables=["dbo.new"],
            removed_tables=["dbo.old"],
        )
        md = report.to_markdown()
        assert "New Tables" in md
        assert "Removed Tables" in md
        assert "`dbo.new`" in md
        assert "`dbo.old`" in md


class TestDetectDrift:
    def test_no_changes(self):
        meta = _make_metadata("dbo", "t1", [("id", "int", False), ("name", "nvarchar", True)])
        inventory = {"dbo.t1": meta}
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
                "name": {"data_type": "nvarchar", "is_nullable": True, "ordinal_position": 1},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        assert not report.has_drift
        assert "dbo.t1" in report.unchanged_tables

    def test_new_table(self):
        inventory = {
            "dbo.t1": _make_metadata("dbo", "t1", [("id", "int", False)]),
            "dbo.t2": _make_metadata("dbo", "t2", [("id", "int", False)]),
        }
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        assert "dbo.t2" in report.new_tables

    def test_removed_table(self):
        inventory = {"dbo.t1": _make_metadata("dbo", "t1", [("id", "int", False)])}
        snapshot = {"tables": {
            "dbo.t1": {
                "schema_name": "dbo",
                "table_name": "t1",
                "row_count": 100,
                "columns": {"id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0}},
            },
            "dbo.t2": {
                "schema_name": "dbo",
                "table_name": "t2",
                "row_count": 50,
                "columns": {"id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0}},
            },
        }}
        report = detect_drift(inventory, snapshot)
        assert "dbo.t2" in report.removed_tables

    def test_column_added(self):
        meta = _make_metadata("dbo", "t1", [
            ("id", "int", False),
            ("name", "nvarchar", True),
            ("email", "nvarchar", True),
        ])
        inventory = {"dbo.t1": meta}
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
                "name": {"data_type": "nvarchar", "is_nullable": True, "ordinal_position": 1},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        assert len(report.changed_tables) == 1
        cc = report.changed_tables[0].column_changes
        added = [c for c in cc if c.change_type == "added"]
        assert len(added) == 1
        assert added[0].column_name == "email"

    def test_column_removed(self):
        meta = _make_metadata("dbo", "t1", [("id", "int", False)])
        inventory = {"dbo.t1": meta}
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
                "name": {"data_type": "nvarchar", "is_nullable": True, "ordinal_position": 1},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        cc = report.changed_tables[0].column_changes
        removed = [c for c in cc if c.change_type == "removed"]
        assert len(removed) == 1
        assert removed[0].column_name == "name"

    def test_type_changed(self):
        meta = _make_metadata("dbo", "t1", [("id", "bigint", False)])
        inventory = {"dbo.t1": meta}
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        cc = report.changed_tables[0].column_changes
        changed = [c for c in cc if c.change_type == "type_changed"]
        assert len(changed) == 1
        assert changed[0].old_value == "int"
        assert changed[0].new_value == "bigint"

    def test_nullability_changed(self):
        meta = _make_metadata("dbo", "t1", [("id", "int", True)])
        inventory = {"dbo.t1": meta}
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        cc = report.changed_tables[0].column_changes
        null_changed = [c for c in cc if c.change_type == "nullability_changed"]
        assert len(null_changed) == 1

    def test_row_count_changed(self):
        meta = _make_metadata("dbo", "t1", [("id", "int", False)], row_count=200)
        inventory = {"dbo.t1": meta}
        snapshot = {"tables": {"dbo.t1": {
            "schema_name": "dbo",
            "table_name": "t1",
            "row_count": 100,
            "columns": {
                "id": {"data_type": "int", "is_nullable": False, "ordinal_position": 0},
            },
        }}}
        report = detect_drift(inventory, snapshot)
        assert len(report.changed_tables) == 1
        assert report.changed_tables[0].change_type == "row_count_changed"
        assert report.changed_tables[0].old_row_count == 100
        assert report.changed_tables[0].new_row_count == 200

    def test_empty_snapshot(self):
        meta = _make_metadata("dbo", "t1", [("id", "int", False)])
        inventory = {"dbo.t1": meta}
        report = detect_drift(inventory, {})
        assert "dbo.t1" in report.new_tables


class TestSnapshotIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        meta = _make_metadata("dbo", "t1", [
            ("id", "int", False),
            ("name", "nvarchar", True),
        ])
        inventory = {"dbo.t1": meta}
        filepath = str(tmp_path / "snapshot.json")
        save_snapshot(inventory, filepath)

        loaded = load_snapshot(filepath)
        assert "tables" in loaded
        assert "dbo.t1" in loaded["tables"]
        assert loaded["tables"]["dbo.t1"]["columns"]["id"]["data_type"] == "int"

    def test_load_nonexistent_file(self):
        result = load_snapshot("/nonexistent/path/snapshot.json")
        assert result == {}
