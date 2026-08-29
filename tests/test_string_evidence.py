"""Tests for string evidence engine."""
from src.models import StringProfile
from src.string_evidence import compute_string_evidence, StringEvidence


def _make_string_profile(
    is_categorical: bool = False,
    is_identifier_like: bool = False,
    distinct_count: int = 10,
    sample_values: list = None,
) -> StringProfile:
    return StringProfile(
        column_name="test_col",
        data_type="nvarchar",
        total_count=100,
        non_null_count=90,
        distinct_count=distinct_count,
        null_ratio=0.1,
        is_categorical=is_categorical,
        categorical_distinct_count=distinct_count if is_categorical else 0,
        is_identifier_like=is_identifier_like,
        avg_length=10.0,
        min_length=1,
        max_length=20,
        contains_numbers=False,
        contains_special_chars=False,
        sample_values=sample_values or [],
    )


class TestComputeStringEvidence:
    def test_both_categorical_overlap(self):
        sp_a = _make_string_profile(
            is_categorical=True,
            distinct_count=5,
            sample_values=["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        )
        sp_b = _make_string_profile(
            is_categorical=True,
            distinct_count=5,
            sample_values=["Alpha", "Beta", "Gamma", "Zeta", "Eta"],
        )

        evidence = compute_string_evidence(sp_a, sp_b)
        assert evidence.is_both_categorical
        assert evidence.categorical_alignment == 0.6  # 3/5 overlap
        assert len(evidence.shared_categories) == 3

    def test_no_categorical(self):
        sp_a = _make_string_profile(
            is_categorical=False,
            sample_values=["val1", "val2", "val3"],
        )
        sp_b = _make_string_profile(
            is_categorical=False,
            sample_values=["val4", "val5", "val6"],
        )

        evidence = compute_string_evidence(sp_a, sp_b)
        assert not evidence.is_both_categorical
        assert evidence.categorical_alignment == 0.0

    def test_token_similarity(self):
        sp_a = _make_string_profile(
            sample_values=["New York", "Los Angeles", "San Francisco"],
        )
        sp_b = _make_string_profile(
            sample_values=["New York", "San Diego", "Los Angeles"],
        )

        evidence = compute_string_evidence(sp_a, sp_b)
        # Tokens: {new, york, los, angeles, san, francisco} vs {new, york, san, diego, los, angeles}
        # Overlap: {new, york, san, los, angeles} = 5, Union: 7
        assert evidence.token_similarity > 0.5

    def test_fuzzy_match(self):
        sp_a = _make_string_profile(
            sample_values=["Apple", "Banana", "Cherry"],
        )
        sp_b = _make_string_profile(
            sample_values=["Aple", "Banana", "Cherry"],
        )

        evidence = compute_string_evidence(sp_a, sp_b)
        # "Aple" should fuzzy-match "Apple" with threshold 0.8
        assert evidence.fuzzy_match_score >= 0.6

    def test_weak_match(self):
        sp_a = _make_string_profile(
            sample_values=["abc123", "def456", "ghi789"],
        )
        sp_b = _make_string_profile(
            sample_values=["xyz000", "uvw111", "rst222"],
        )

        evidence = compute_string_evidence(sp_a, sp_b)
        assert evidence.weak_match

    def test_empty_values(self):
        sp_a = _make_string_profile(sample_values=[])
        sp_b = _make_string_profile(sample_values=[])

        evidence = compute_string_evidence(sp_a, sp_b)
        assert evidence.categorical_alignment == 0.0
        assert evidence.token_similarity == 0.0
        assert evidence.fuzzy_match_score == 0.0

    def test_string_score_calculation(self):
        sp_a = _make_string_profile(
            is_categorical=True,
            sample_values=["A", "B", "C", "D", "E"],
        )
        sp_b = _make_string_profile(
            is_categorical=True,
            sample_values=["A", "B", "C", "D", "E"],
        )

        evidence = compute_string_evidence(sp_a, sp_b)
        assert evidence.string_score > 0.5


class TestStringEvidenceDataclass:
    def test_default_values(self):
        se = StringEvidence()
        assert se.categorical_alignment == 0.0
        assert se.string_score == 0.0
        assert not se.weak_match

    def test_weak_match_downgrades_score(self):
        se = StringEvidence(
            categorical_alignment=0.5,
            token_similarity=0.4,
            normalization_overlap=0.3,
            weak_match=True,
        )
        # Score should be halved due to weak_match
        assert se.string_score < 0.3