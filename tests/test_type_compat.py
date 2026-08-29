"""Tests for type compatibility evaluator."""
from src.type_compat import evaluate_type_compatibility, filter_compatible_candidates, TypeDecision
from src.candidates import Candidate


class TestEvaluateTypeCompatibility:
    def test_same_family_compatible(self):
        decision = evaluate_type_compatibility("int", "bigint")
        assert decision.compatible
        assert decision.risk is None
        assert decision.canonical_a == "numeric"
        assert decision.canonical_b == "numeric"

    def test_string_compatible(self):
        decision = evaluate_type_compatibility("nvarchar(255)", "varchar(100)")
        assert decision.compatible
        assert decision.risk is None

    def test_datetime_compatible(self):
        decision = evaluate_type_compatibility("datetime2", "date")
        assert decision.compatible
        assert decision.risk is None

    def test_incompatible_strict(self):
        decision = evaluate_type_compatibility("int", "nvarchar", strict=True)
        assert not decision.compatible
        assert decision.risk == "incompatible"

    def test_cast_required_relaxed(self):
        decision = evaluate_type_compatibility("int", "nvarchar", strict=False)
        assert decision.compatible
        assert decision.risk == "cast_required"

    def test_boolean_compatible(self):
        decision = evaluate_type_compatibility("bit", "bit")
        assert decision.compatible

    def test_unknown_type(self):
        decision = evaluate_type_compatibility("custom_type_a", "custom_type_b")
        # Both unknown should be same canonical family
        assert decision.compatible
        assert decision.canonical_a == "unknown"
        assert decision.canonical_b == "unknown"


class TestFilterCompatibleCandidates:
    def test_filters_incompatible(self):
        candidates = [
            Candidate(
                source_table="t1", source_column="id", source_type="int",
                target_table="t2", target_column="id", target_type="bigint",
                name_score=1.0, match_type="exact_name",
            ),
            Candidate(
                source_table="t1", source_column="name", source_type="int",
                target_table="t2", target_column="code", target_type="nvarchar",
                name_score=0.8, match_type="fuzzy",
            ),
        ]
        compatible, rejected = filter_compatible_candidates(candidates, {"thresholds": {"type_compatibility": "strict"}})
        assert len(compatible) == 1
        assert len(rejected) == 1
        assert compatible[0].source_column == "id"

    def test_relaxed_allows_cast(self):
        candidates = [
            Candidate(
                source_table="t1", source_column="id", source_type="int",
                target_table="t2", target_column="id_str", target_type="nvarchar",
                name_score=0.9, match_type="fuzzy",
            ),
        ]
        compatible, rejected = filter_compatible_candidates(candidates, {"thresholds": {"type_compatibility": "relaxed"}})
        assert len(compatible) == 1
        assert len(rejected) == 0
        assert compatible[0].type_decision.risk == "cast_required"

    def test_attaches_type_decision(self):
        candidates = [
            Candidate(
                source_table="t1", source_column="id", source_type="int",
                target_table="t2", target_column="id", target_type="bigint",
                name_score=1.0, match_type="exact_name",
            ),
        ]
        compatible, rejected = filter_compatible_candidates(candidates)
        assert hasattr(compatible[0], "type_decision")
        assert isinstance(compatible[0].type_decision, TypeDecision)