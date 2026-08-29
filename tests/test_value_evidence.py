"""Tests for value evidence engine."""
from src.models import ColumnProfile
from src.value_evidence import compute_value_evidence, ValueEvidence


def _make_profile(name: str, data_type: str, row_count: int, top_values: list,
                  distinct_count: int = 0, null_count: int = 0) -> ColumnProfile:
    non_null = row_count - null_count
    return ColumnProfile(
        name=name,
        data_type=data_type,
        row_count=row_count,
        non_null_count=non_null,
        null_count=null_count,
        null_ratio=null_count / row_count if row_count > 0 else 0.0,
        distinct_count=distinct_count,
        distinct_ratio=distinct_count / non_null if non_null > 0 else 0.0,
        uniqueness_ratio=distinct_count / row_count if row_count > 0 else 0.0,
        top_values=top_values,
    )


class TestComputeValueEvidence:
    def test_exact_overlap(self):
        src = _make_profile("id", "int", 100, [1, 2, 3, 4, 5], distinct_count=5)
        tgt = _make_profile("id", "int", 100, [1, 2, 3, 4, 5], distinct_count=5)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.exact_overlap_count == 5
        assert evidence.exact_overlap_ratio == 1.0
        assert evidence.jaccard == 1.0

    def test_partial_overlap(self):
        src = _make_profile("id", "int", 100, [1, 2, 3, 4, 5], distinct_count=5)
        tgt = _make_profile("id", "int", 100, [3, 4, 5, 6, 7], distinct_count=5)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.exact_overlap_count == 3
        assert evidence.exact_overlap_ratio == 0.6
        # Jaccard = 3/7 = 0.4286
        assert 0.4 < evidence.jaccard < 0.45

    def test_no_overlap(self):
        src = _make_profile("id", "int", 100, [1, 2, 3], distinct_count=3)
        tgt = _make_profile("id", "int", 100, [4, 5, 6], distinct_count=3)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.exact_overlap_count == 0
        assert evidence.exact_overlap_ratio == 0.0
        assert evidence.jaccard == 0.0

    def test_containment(self):
        src = _make_profile("id", "int", 100, [1, 2], distinct_count=2)
        tgt = _make_profile("id", "int", 100, [1, 2, 3, 4, 5], distinct_count=5)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.containment_source_to_target == 0.4  # 2/5
        assert evidence.containment_target_to_source == 1.0  # 2/2
        assert "target_contained_in_source" in evidence.confidence_flags

    def test_string_normalized_overlap(self):
        src = _make_profile("name", "nvarchar", 100, ["Alpha", "Beta", "GAMMA"], distinct_count=3)
        tgt = _make_profile("name", "nvarchar", 100, ["alpha", "beta", "delta"], distinct_count=3)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.exact_overlap_count == 0  # case-sensitive
        assert evidence.normalized_overlap_count == 2  # alpha, beta match after lower
        assert abs(evidence.normalized_overlap_ratio - 2 / 3) < 0.001

    def test_empty_values(self):
        src = _make_profile("id", "int", 100, [], distinct_count=0)
        tgt = _make_profile("id", "int", 100, [1, 2, 3], distinct_count=3)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.exact_overlap_count == 0
        assert evidence.value_score == 0.0

    def test_value_score_calculation(self):
        src = _make_profile("id", "int", 100, [1, 2, 3, 4, 5], distinct_count=5)
        tgt = _make_profile("id", "int", 100, [1, 2, 3, 4, 5], distinct_count=5)

        evidence = compute_value_evidence(src, tgt)
        assert evidence.value_score > 0.8  # High overlap should score well


class TestValueEvidenceDataclass:
    def test_default_values(self):
        ve = ValueEvidence()
        assert ve.exact_overlap_count == 0
        assert ve.mode == "sampled"
        assert ve.value_score == 0.0

    def test_mode_property(self):
        ve = ValueEvidence(mode="full")
        assert ve.mode == "full"