"""Tests for type canonicalization."""
from src.types import (
    canonicalize_type,
    are_types_compatible,
    type_compatibility_decision,
    CanonicalType,
    _strip_type_modifiers,
    FALLBACK_TYPE_MAP,
)


class TestStripTypeModifiers:
    def test_strips_length(self):
        assert _strip_type_modifiers("nvarchar(255)") == "nvarchar"

    def test_strips_precision_scale(self):
        assert _strip_type_modifiers("decimal(18,2)") == "decimal"

    def test_unchanged_without_modifiers(self):
        assert _strip_type_modifiers("int") == "int"

    def test_handles_spaces(self):
        assert _strip_type_modifiers("  nvarchar (max) ") == "nvarchar"


class TestCanonicalizeType:
    def test_numeric_types(self):
        for t in ["int", "bigint", "smallint", "tinyint", "decimal", "numeric",
                  "float", "real", "money", "smallmoney"]:
            assert canonicalize_type(t) == CanonicalType.NUMERIC, f"Failed for {t}"

    def test_string_types(self):
        for t in ["varchar", "nvarchar", "char", "nchar", "text", "ntext"]:
            assert canonicalize_type(t) == CanonicalType.STRING, f"Failed for {t}"

    def test_datetime_types(self):
        for t in ["date", "time", "datetime", "datetime2", "smalldatetime", "datetimeoffset"]:
            assert canonicalize_type(t) == CanonicalType.DATETIME, f"Failed for {t}"

    def test_boolean(self):
        assert canonicalize_type("bit") == CanonicalType.BOOLEAN

    def test_binary(self):
        assert canonicalize_type("binary") == CanonicalType.BINARY
        assert canonicalize_type("varbinary") == CanonicalType.BINARY

    def test_spatial(self):
        assert canonicalize_type("geometry") == CanonicalType.SPATIAL
        assert canonicalize_type("geography") == CanonicalType.SPATIAL

    def test_unknown_type(self):
        assert canonicalize_type("my_custom_type") == CanonicalType.UNKNOWN

    def test_handles_modifiers(self):
        assert canonicalize_type("nvarchar(255)") == CanonicalType.STRING
        assert canonicalize_type("decimal(18,2)") == CanonicalType.NUMERIC

    def test_case_insensitive(self):
        assert canonicalize_type("NVARCHAR") == CanonicalType.STRING
        assert canonicalize_type("Int") == CanonicalType.NUMERIC


class TestAreTypesCompatible:
    def test_same_type(self):
        assert are_types_compatible("int", "int") is True

    def test_same_family(self):
        assert are_types_compatible("int", "bigint") is True
        assert are_types_compatible("varchar", "nvarchar") is True

    def test_different_family_strict(self):
        assert are_types_compatible("int", "varchar", strict=True) is False

    def test_different_family_relaxed(self):
        assert are_types_compatible("int", "varchar", strict=False) is True

    def test_incompatible_types(self):
        assert are_types_compatible("int", "date") is False


class TestTypeCompatibilityDecision:
    def test_compatible_decision(self):
        d = type_compatibility_decision("int", "bigint")
        assert d["compatible"] is True
        assert d["risk"] is None

    def test_cast_required_decision(self):
        d = type_compatibility_decision("int", "varchar", strict=False)
        assert d["compatible"] is True
        assert d["risk"] == "cast_required"

    def test_incompatible_decision(self):
        d = type_compatibility_decision("int", "date")
        assert d["compatible"] is False
        assert d["risk"] == "incompatible"


class TestFallbackMapCoverage:
    def test_all_fallback_types_exist(self):
        """Every type in FALLBACK_TYPE_MAP should be canonicalizable."""
        for native_type in FALLBACK_TYPE_MAP:
            assert canonicalize_type(native_type) != CanonicalType.UNKNOWN