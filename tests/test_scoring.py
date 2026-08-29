"""Tests for scoring engine."""
from src.models import ColumnProfile
from src.scoring import compute_score, ScoreBreakdown, _classify_band
from src.type_compat import TypeDecision, evaluate_type_compatibility


def _make_profile(distinct_count: int = 100, null_count: int = 0, row_count: int = 100) -> ColumnProfile:
    non_null = row_count - null_count
    return ColumnProfile(
        name="id",
        data_type="int",
        row_count=row_count,
        non_null_count=non_null,
        null_count=null_count,
        null_ratio=null_count / row_count if row_count > 0 else 0.0,
        distinct_count=distinct_count,
        distinct_ratio=distinct_count / non_null if non_null > 0 else 0.0,
        uniqueness_ratio=distinct_count / row_count if row_count > 0 else 0.0,
    )


def _make_type_decision(compatible: bool = True, risk: str = None) -> TypeDecision:
    return TypeDecision(
        compatible=compatible,
        canonical_a="numeric",
        canonical_b="numeric",
        risk=risk,
        reason="test",
    )


class TestClassifyBand:
    def test_high(self):
        assert _classify_band(0.90) == "high"

    def test_medium(self):
        assert _classify_band(0.75) == "medium"

    def test_low(self):
        assert _classify_band(0.60) == "low"

    def test_reject(self):
        assert _classify_band(0.40) == "reject"

    def test_boundary_high(self):
        assert _classify_band(0.85) == "high"

    def test_boundary_medium(self):
        assert _classify_band(0.70) == "medium"

    def test_boundary_low(self):
        assert _classify_band(0.50) == "low"


class TestComputeScore:
    def test_perfect_score(self):
        decision = _make_type_decision(compatible=True, risk=None)
        src = _make_profile(distinct_count=100, null_count=0)
        tgt = _make_profile(distinct_count=100, null_count=0)

        breakdown = compute_score(
            name_score=1.0,
            type_decision=decision,
            value_score=1.0,
            source_profile=src,
            target_profile=tgt,
        )
        # All components should be 1.0
        assert breakdown.confidence == 1.0
        assert breakdown.confidence_band == "high"

    def test_low_score(self):
        decision = _make_type_decision(compatible=False, risk="incompatible")
        src = _make_profile(distinct_count=10, null_count=50)
        tgt = _make_profile(distinct_count=90, null_count=0)

        breakdown = compute_score(
            name_score=0.0,
            type_decision=decision,
            value_score=0.0,
            source_profile=src,
            target_profile=tgt,
        )
        assert breakdown.confidence < 0.5
        assert breakdown.confidence_band == "reject"

    def test_cast_required_penalty(self):
        decision = _make_type_decision(compatible=True, risk="cast_required")
        src = _make_profile(distinct_count=100, null_count=0)
        tgt = _make_profile(distinct_count=100, null_count=0)

        breakdown = compute_score(
            name_score=1.0,
            type_decision=decision,
            value_score=1.0,
            source_profile=src,
            target_profile=tgt,
        )
        # type_score should be 0.7 for cast_required
        assert breakdown.type_score == 0.7
        # Overall should be medium (not high due to type penalty)
        assert breakdown.confidence_band in ("medium", "high")

    def test_uniqueness_mismatch(self):
        decision = _make_type_decision(compatible=True, risk=None)
        src = _make_profile(distinct_count=100, null_count=0)  # uniqueness = 1.0
        tgt = _make_profile(distinct_count=10, null_count=0, row_count=100)  # uniqueness = 0.1

        breakdown = compute_score(
            name_score=1.0,
            type_decision=decision,
            value_score=1.0,
            source_profile=src,
            target_profile=tgt,
        )
        # uniqueness_score should be penalized (diff = 0.9, score = 0.1)
        assert breakdown.uniqueness_score < 0.2

    def test_null_pattern_mismatch(self):
        decision = _make_type_decision(compatible=True, risk=None)
        src = _make_profile(null_count=0)     # null_ratio = 0.0
        tgt = _make_profile(null_count=50)    # null_ratio = 0.5

        breakdown = compute_score(
            name_score=1.0,
            type_decision=decision,
            value_score=1.0,
            source_profile=src,
            target_profile=tgt,
        )
        assert breakdown.null_score < 1.0

    def test_custom_weights(self):
        decision = _make_type_decision(compatible=True, risk=None)
        src = _make_profile(distinct_count=100, null_count=0)
        tgt = _make_profile(distinct_count=100, null_count=0)

        custom_weights = {
            "name": 0.0,
            "type": 0.0,
            "value": 1.0,
            "uniqueness": 0.0,
            "null": 0.0,
        }
        breakdown = compute_score(
            name_score=0.5,
            type_decision=decision,
            value_score=0.8,
            source_profile=src,
            target_profile=tgt,
            weights=custom_weights,
        )
        assert breakdown.confidence == 0.8

    def test_missing_profiles(self):
        decision = _make_type_decision(compatible=True, risk=None)

        breakdown = compute_score(
            name_score=0.8,
            type_decision=decision,
            value_score=0.6,
            source_profile=None,
            target_profile=None,
        )
        # uniqueness and null should be 0.0 when profiles missing
        assert breakdown.uniqueness_score == 0.0
        assert breakdown.null_score == 0.0


class TestScoreBreakdown:
    def test_default_values(self):
        bd = ScoreBreakdown()
        assert bd.confidence == 0.0
        assert bd.confidence_band == "reject"