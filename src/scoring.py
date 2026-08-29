"""Precision-biased scoring engine.

Computes confidence scores for candidate relationships using a
configurable weighted formula:

    score = 0.15 * name_score
          + 0.30 * type_score
          + 0.40 * value_score
          + 0.10 * uniqueness_alignment
          + 0.05 * null_pattern_score

Results are banded into High/Medium/Low/Reject.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.models import ColumnProfile
from src.type_compat import TypeDecision


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a confidence score."""
    name_score: float = 0.0
    type_score: float = 0.0
    value_score: float = 0.0
    uniqueness_score: float = 0.0
    null_score: float = 0.0
    confidence: float = 0.0
    confidence_band: str = "reject"
    weights: Dict[str, float] = field(default_factory=dict)


# Default scoring weights
_DEFAULT_WEIGHTS = {
    "name": 0.15,
    "type": 0.30,
    "value": 0.40,
    "uniqueness": 0.10,
    "null": 0.05,
}

# Confidence bands
_BANDS = {
    "high": 0.85,
    "medium": 0.70,
    "low": 0.50,
}


def _compute_type_score(decision: TypeDecision) -> float:
    """Compute type component score (0.0 - 1.0).

    1.0 = same canonical family, no risk
    0.7 = compatible with cast_required risk
    0.0 = incompatible
    """
    if not decision.compatible:
        return 0.0
    if decision.risk == "cast_required":
        return 0.7
    return 1.0


def _compute_uniqueness_score(
    source_profile: Optional[ColumnProfile],
    target_profile: Optional[ColumnProfile],
) -> float:
    """Compute uniqueness alignment score (0.0 - 1.0).

    Measures how well the uniqueness ratios of the two columns align.
    High score when both are highly unique or both are low uniqueness.
    """
    if not source_profile or not target_profile:
        return 0.0

    src_uniq = source_profile.uniqueness_ratio
    tgt_uniq = target_profile.uniqueness_ratio

    # Perfect alignment
    if src_uniq == tgt_uniq:
        return 1.0

    # Penalize large differences
    diff = abs(src_uniq - tgt_uniq)
    return max(0.0, 1.0 - diff)


def _compute_null_pattern_score(
    source_profile: Optional[ColumnProfile],
    target_profile: Optional[ColumnProfile],
) -> float:
    """Compute null pattern alignment score (0.0 - 1.0).

    High score when both columns have similar null ratios.
    """
    if not source_profile or not target_profile:
        return 0.0

    src_null = source_profile.null_ratio
    tgt_null = target_profile.null_ratio

    diff = abs(src_null - tgt_null)
    return max(0.0, 1.0 - diff)


def _classify_band(confidence: float) -> str:
    """Classify a confidence score into a band."""
    if confidence >= _BANDS["high"]:
        return "high"
    if confidence >= _BANDS["medium"]:
        return "medium"
    if confidence >= _BANDS["low"]:
        return "low"
    return "reject"


def compute_score(
    name_score: float,
    type_decision: TypeDecision,
    value_score: float,
    source_profile: Optional[ColumnProfile],
    target_profile: Optional[ColumnProfile],
    weights: Optional[Dict[str, float]] = None,
) -> ScoreBreakdown:
    """Compute a full confidence score for a candidate relationship.

    Args:
        name_score: Name similarity score (0.0 - 1.0).
        type_decision: Type compatibility decision.
        value_score: Value overlap score (0.0 - 1.0).
        source_profile: Profile of the source column.
        target_profile: Profile of the target column.
        weights: Optional custom weights (defaults to _DEFAULT_WEIGHTS).

    Returns:
        ScoreBreakdown with all components and final confidence.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS

    # Compute component scores
    t_score = _compute_type_score(type_decision)
    u_score = _compute_uniqueness_score(source_profile, target_profile)
    n_score = _compute_null_pattern_score(source_profile, target_profile)

    # Apply weights
    confidence = (
        weights["name"] * name_score
        + weights["type"] * t_score
        + weights["value"] * value_score
        + weights["uniqueness"] * u_score
        + weights["null"] * n_score
    )

    band = _classify_band(confidence)

    return ScoreBreakdown(
        name_score=round(name_score, 4),
        type_score=round(t_score, 4),
        value_score=round(value_score, 4),
        uniqueness_score=round(u_score, 4),
        null_score=round(n_score, 4),
        confidence=round(confidence, 4),
        confidence_band=band,
        weights=dict(weights),
    )


def compute_score_batch(
    candidates,
    type_decisions: Dict[tuple, TypeDecision],
    value_scores: Dict[tuple, float],
    source_profiles: Dict[tuple, ColumnProfile],
    target_profiles: Dict[tuple, ColumnProfile],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[tuple, ScoreBreakdown]:
    """Compute scores for all candidates in batch.

    Args:
        candidates: List of Candidate objects.
        type_decisions: Mapping of (src_table, src_col, tgt_table, tgt_col) -> TypeDecision.
        value_scores: Mapping of (src_table, src_col, tgt_table, tgt_col) -> value_score float.
        source_profiles: Mapping of (table, col) -> ColumnProfile.
        target_profiles: Mapping of (table, col) -> ColumnProfile.
        config: Configuration dict with optional custom weights.

    Returns:
        Mapping of (src_table, src_col, tgt_table, tgt_col) -> ScoreBreakdown.
    """
    if config is None:
        config = {}

    thresholds = config.get("thresholds", {})
    weights = thresholds.get("scoring_weights", _DEFAULT_WEIGHTS)

    results: Dict[tuple, ScoreBreakdown] = {}

    for c in candidates:
        key = (c.source_table, c.source_column, c.target_table, c.target_column)
        decision = type_decisions.get(key)
        v_score = value_scores.get(key, 0.0)
        src_prof = source_profiles.get((c.source_table, c.source_column))
        tgt_prof = target_profiles.get((c.target_table, c.target_column))

        if decision is None:
            continue

        breakdown = compute_score(
            name_score=c.name_score,
            type_decision=decision,
            value_score=v_score,
            source_profile=src_prof,
            target_profile=tgt_prof,
            weights=weights,
        )
        results[key] = breakdown

    return results