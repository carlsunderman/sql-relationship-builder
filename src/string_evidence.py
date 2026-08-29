"""String evidence engine.

Analyzes string-type candidates for categorical alignment, token similarity,
and fuzzy matching. Prioritizes precision; weak matches are downgraded.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set

from src.models import StringProfile


@dataclass
class StringEvidence:
    """String-specific evidence for a candidate relationship."""
    categorical_alignment: float = 0.0
    token_similarity: float = 0.0
    normalization_overlap: float = 0.0
    fuzzy_match_score: float = 0.0
    is_both_categorical: bool = False
    is_both_identifier: bool = False
    shared_categories: List[str] = field(default_factory=list)
    weak_match: bool = False

    @property
    def string_score(self) -> float:
        """Derive a normalized string evidence score.

        Weighted combination of categorical alignment, token similarity,
        and normalization overlap.
        """
        score = (
            0.45 * self.categorical_alignment
            + 0.30 * self.token_similarity
            + 0.25 * self.normalization_overlap
        )
        # Downgrade weak matches
        if self.weak_match:
            score *= 0.5
        return round(score, 4)


def _tokenize(value: str) -> Set[str]:
    """Split a string into tokens (lowercase, alphanumeric)."""
    return set(value.lower().split())


def _compute_token_similarity(values_a: List[str], values_b: List[str]) -> float:
    """Compute token-level similarity between two sets of string values.

    Uses the union of tokens from both sets and measures overlap.
    """
    tokens_a: Set[str] = set()
    tokens_b: Set[str] = set()

    for v in values_a:
        tokens_a.update(str(v).lower().split())
    for v in values_b:
        tokens_b.update(str(v).lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    overlap = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(overlap) / len(union) if union else 0.0


def _compute_fuzzy_match(values_a: List[str], values_b: List[str], threshold: float = 0.8) -> float:
    """Compute fuzzy match score between two sets of string values.

    For each value in the smaller set, check if any value in the larger set
    has a SequenceMatcher ratio >= threshold.

    Returns the fraction of values that have a fuzzy match.
    """
    if not values_a or not values_b:
        return 0.0

    # Compare smaller set against larger set
    if len(values_a) > len(values_b):
        values_a, values_b = values_b, values_a

    matches = 0
    for va in values_a:
        va_str = str(va).lower().strip()
        for vb in values_b:
            vb_str = str(vb).lower().strip()
            if SequenceMatcher(None, va_str, vb_str).ratio() >= threshold:
                matches += 1
                break

    return matches / len(values_a) if values_a else 0.0


def _compute_normalization_overlap(values_a: List[str], values_b: List[str]) -> float:
    """Compute overlap after normalization (lowercase, whitespace trim)."""
    if not values_a or not values_b:
        return 0.0

    norm_a = set(str(v).lower().strip() for v in values_a)
    norm_b = set(str(v).lower().strip() for v in values_b)

    overlap = norm_a & norm_b
    min_size = min(len(norm_a), len(norm_b))
    return len(overlap) / min_size if min_size > 0 else 0.0


def _compute_categorical_alignment(
    profile_a: StringProfile,
    profile_b: StringProfile,
    sample_values_a: List[str],
    sample_values_b: List[str],
) -> float:
    """Compute categorical alignment between two string columns.

    If both columns are categorical, measures how much their value sets overlap.
    """
    if not profile_a.is_categorical and not profile_b.is_categorical:
        return 0.0

    if profile_a.is_categorical and profile_b.is_categorical:
        # Both categorical: measure direct overlap of value sets
        values_a = set(str(v).lower().strip() for v in sample_values_a)
        values_b = set(str(v).lower().strip() for v in sample_values_b)

        if not values_a or not values_b:
            return 0.0

        overlap = values_a & values_b
        min_size = min(len(values_a), len(values_b))
        return len(overlap) / min_size if min_size > 0 else 0.0

    # One categorical, one not: check if the categorical values appear in the other
    cat_profile = profile_a if profile_a.is_categorical else profile_b
    other_values = (sample_values_b if profile_a.is_categorical else sample_values_a)

    cat_values = set(str(v).lower().strip() for v in (
        sample_values_a if profile_a.is_categorical else sample_values_b
    ))
    other_set = set(str(v).lower().strip() for v in other_values)

    if not cat_values or not other_set:
        return 0.0

    overlap = cat_values & other_set
    return len(overlap) / len(cat_values) if cat_values else 0.0


def compute_string_evidence(
    profile_a: StringProfile,
    profile_b: StringProfile,
    sample_values_a: Optional[List[str]] = None,
    sample_values_b: Optional[List[str]] = None,
    fuzzy_threshold: float = 0.8,
) -> StringEvidence:
    """Compute string-specific evidence between two string columns.

    Args:
        profile_a: String profile of column A.
        profile_b: String profile of column B.
        sample_values_a: Sample values from column A (defaults to profile.sample_values).
        sample_values_b: Sample values from column B (defaults to profile.sample_values).
        fuzzy_threshold: Minimum SequenceMatcher ratio for fuzzy matching.

    Returns:
        StringEvidence with alignment and similarity metrics.
    """
    vals_a = sample_values_a or profile_a.sample_values
    vals_b = sample_values_b or profile_b.sample_values

    # Categorical alignment
    cat_alignment = _compute_categorical_alignment(profile_a, profile_b, vals_a, vals_b)

    # Token similarity
    token_sim = _compute_token_similarity(vals_a, vals_b)

    # Normalization overlap
    norm_overlap = _compute_normalization_overlap(vals_a, vals_b)

    # Fuzzy match
    fuzzy = _compute_fuzzy_match(vals_a, vals_b, fuzzy_threshold)

    # Shared categories
    if profile_a.is_categorical and profile_b.is_categorical:
        set_a = set(str(v).lower().strip() for v in vals_a)
        set_b = set(str(v).lower().strip() for v in vals_b)
        shared = list(set_a & set_b)[:20]
    else:
        shared = []

    # Weak match detection
    weak = (
        cat_alignment < 0.1
        and token_sim < 0.1
        and fuzzy < 0.3
    )

    return StringEvidence(
        categorical_alignment=round(cat_alignment, 4),
        token_similarity=round(token_sim, 4),
        normalization_overlap=round(norm_overlap, 4),
        fuzzy_match_score=round(fuzzy, 4),
        is_both_categorical=profile_a.is_categorical and profile_b.is_categorical,
        is_both_identifier=profile_a.is_identifier_like and profile_b.is_identifier_like,
        shared_categories=shared,
        weak_match=weak,
    )


def compute_string_evidence_batch(
    string_profiles: Dict[str, Dict[str, StringProfile]],
    candidates,
    fuzzy_threshold: float = 0.8,
) -> Dict[tuple, StringEvidence]:
    """Compute string evidence for all candidates in batch.

    Args:
        string_profiles: Nested dict {table_key: {col_name: StringProfile}}.
        candidates: List of Candidate objects.
        fuzzy_threshold: Fuzzy matching threshold.

    Returns:
        Mapping of (source_table, source_col, target_table, target_col) -> StringEvidence.
    """
    results: Dict[tuple, StringEvidence] = {}

    for c in candidates:
        sp_a = string_profiles.get(c.source_table, {}).get(c.source_column)
        sp_b = string_profiles.get(c.target_table, {}).get(c.target_column)

        if sp_a and sp_b:
            evidence = compute_string_evidence(sp_a, sp_b, fuzzy_threshold=fuzzy_threshold)
            results[(c.source_table, c.source_column, c.target_table, c.target_column)] = evidence
        else:
            results[(c.source_table, c.source_column, c.target_table, c.target_column)] = StringEvidence()

    return results