"""Tests for config validation."""
import json
import os
import tempfile

import jsonschema
import pytest
import yaml


_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def defaults() -> dict:
    path = os.path.join(_CONFIG_DIR, "defaults.yaml")
    if not os.path.exists(path):
        pytest.skip("config/defaults.yaml not found")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def schema() -> dict:
    path = os.path.join(_CONFIG_DIR, "schema.json")
    if not os.path.exists(path):
        pytest.skip("config/schema.json not found")
    with open(path) as f:
        return json.load(f)


class TestConfigValidation:
    def test_defaults_validates_against_schema(self, defaults: dict, schema: dict):
        """P1-T2: defaults.yaml must validate against schema.json"""
        jsonschema.validate(instance=defaults, schema=schema)

    def test_thresholds_have_required_keys(self, defaults: dict):
        required = [
            "name_similarity_min", "type_compatibility", "value_overlap_min",
            "value_overlap_ratio", "jaccard_min", "confidence_high",
            "confidence_medium", "confidence_low", "string_categorical_distinct_max",
        ]
        for key in required:
            assert key in defaults.get("thresholds", {}), f"Missing threshold: {key}"

    def test_aliases_are_lists(self, defaults: dict):
        for alias, values in defaults.get("aliases", {}).items():
            assert isinstance(values, list), f"Alias {alias} is not a list"
            assert len(values) > 0, f"Alias {alias} is empty"

    def test_thresholds_in_range(self, defaults: dict):
        thresholds = defaults.get("thresholds", {})
        for key in ["name_similarity_min", "value_overlap_ratio", "jaccard_min",
                     "confidence_high", "confidence_medium", "confidence_low"]:
            val = thresholds.get(key)
            assert val is not None, f"Missing threshold: {key}"
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range [0, 1]"

    def test_type_compatibility_valid(self, defaults: dict):
        tc = defaults.get("thresholds", {}).get("type_compatibility")
        assert tc in ("strict", "relaxed"), f"Invalid type_compatibility: {tc}"

    def test_profiling_config_shape(self, defaults: dict):
        profiling = defaults.get("profiling", {})
        assert "mode_a_max_rows" in profiling
        assert "mode_b_string_cardinality" in profiling
        assert "mode_c_sample_sizes" in profiling
        assert isinstance(profiling["mode_c_sample_sizes"], list)

    def test_exclusions_are_lists(self, defaults: dict):
        exclusions = defaults.get("exclusions", {})
        assert isinstance(exclusions.get("column_patterns", None), list)
        assert isinstance(exclusions.get("table_patterns", None), list)