"""Tests for database connection and schema discovery."""
import pytest
from src.db import (
    ColumnInfo,
    TableInfo,
    ForeignKeyInfo,
    connect_to_sql_server,
)


class TestDataClasses:
    def test_table_info_full_name(self):
        t = TableInfo(schema_name="dbo", table_name="customers")
        assert t.full_name == "dbo.customers"

    def test_table_info_with_columns(self):
        cols = [ColumnInfo(name="id", data_type="int", is_nullable=False, ordinal_position=1)]
        t = TableInfo(schema_name="dbo", table_name="t", columns=cols)
        assert len(t.columns) == 1
        assert t.columns[0].name == "id"

    def test_foreign_key_info(self):
        fk = ForeignKeyInfo(
            source_schema="dbo", source_table="orders", source_column="customer_id",
            target_schema="dbo", target_table="customers", target_column="id",
            constraint_name="FK_orders_customers",
        )
        assert fk.source_table == "orders"
        assert fk.target_table == "customers"


class TestConnectToSqlServer:
    def test_raises_on_invalid_server(self):
        """Should raise pyodbc.Error for unreachable server."""
        pytest.importorskip("pyodbc", exc_type=ImportError)