"""Value evidence engine.

Computes value overlap, containment, and Jaccard metrics between
candidate column pairs using pushdown SQL (no full-table pandas load).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models import ColumnProfile, TableProfile


@dataclass
class ValueEvidence:
    """Value overlap evidence for a candidate relationship."""
    exact_overlap_count: int = 0
    exact_overlap_ratio: float = 0.0
    normalized_overlap_count: int = 0
    normalized_overlap_ratio: float = 0.0
    jaccard: float = 0.0
    containment_source_to_target: float = 0.0
    containment_target_to_source: float = 0.0
    mode: str = "sampled"  # "full" or "sampled"
    sample_size: int = 0
    common_values: List[Any] = field(default_factory=list)
    confidence_flags: List[str] = field(default_factory=list)

    @property
    def value_score(self) -> float:
        """Derive a normalized value score from the evidence.

        Uses the best of overlap ratio and containment, weighted by Jaccard.
        """
        best_containment = max(
            self.containment_source_to_target,
            self.containment_target_to_source,
        )
        # Weighted combination
        return (
            0.40 * self.exact_overlap_ratio
            + 0.35 * best_containment
            + 0.25 * self.jaccard
        )


def _get_value_set(
    profile: ColumnProfile,
    sample_size: int,
) -> Tuple[Set[Any], int, str]:
    """Extract a value set from a column profile.

    Uses top_values from pushdown profiling. For large tables,
    this is a sampled set. For small tables, it may be the full set.

    Args:
        profile: Column profile with top_values.
        sample_size: Expected sample size.

    Returns:
        Tuple of (value_set, set_size, mode).
    """
    values = profile.top_values
    value_set = set(values)
    mode = "full" if profile.row_count <= sample_size else "sampled"
    return value_set, len(value_set), mode


def compute_value_evidence(
    source_profile: ColumnProfile,
    target_profile: ColumnProfile,
    sample_size: int = 100,
) -> ValueEvidence:
    """Compute value overlap evidence between two columns.

    Uses the top_values from pushdown profiling to estimate overlap.
    For small tables (<= sample_size), this is exact. For large tables,
    it's a sampled estimate.

    Args:
        source_profile: Profile of the source column.
        target_profile: Profile of the target column.
        sample_size: Threshold for full vs sampled mode.

    Returns:
        ValueEvidence with overlap metrics.
    """
    src_set, src_size, src_mode = _get_value_set(source_profile, sample_size)
    tgt_set, tgt_size, tgt_mode = _get_value_set(target_profile, sample_size)

    mode = "full" if src_mode == "full" and tgt_mode == "full" else "sampled"

    if not src_set or not tgt_set:
        return ValueEvidence(mode=mode, sample_size=sample_size)

    # Exact overlap
    overlap = src_set & tgt_set
    overlap_count = len(overlap)
    min_size = min(src_size, tgt_size)
    max_size = max(src_size, tgt_size)

    exact_overlap_ratio = overlap_count / min_size if min_size > 0 else 0.0

    # Jaccard similarity
    union = src_set | tgt_set
    jaccard = overlap_count / len(union) if len(union) > 0 else 0.0

    # Containment ratios
    containment_s_to_t = overlap_count / tgt_size if tgt_size > 0 else 0.0
    containment_t_to_s = overlap_count / src_size if src_size > 0 else 0.0

    # Normalized overlap (case-insensitive for strings)
    normalized_overlap = 0
    normalized_ratio = 0.0
    if source_profile.data_type.lower().startswith(("nvar", "var", "char", "nchar", "text")):
        src_norm = set(str(v).lower().strip() for v in src_set)
        tgt_norm = set(str(v).lower().strip() for v in tgt_set)
        normalized_overlap = len(src_norm & tgt_norm)
        normalized_ratio = normalized_overlap / min(len(src_norm), len(tgt_norm)) if min(len(src_norm), len(tgt_norm)) > 0 else 0.0

    # Confidence flags
    flags = []
    if containment_s_to_t > 0.95:
        flags.append("source_contained_in_target")
    if containment_t_to_s > 0.95:
        flags.append("target_contained_in_source")
    if jaccard > 0.8:
        flags.append("high_jaccard")
    if mode == "sampled":
        flags.append("sampled_estimate")

    return ValueEvidence(
        exact_overlap_count=overlap_count,
        exact_overlap_ratio=round(exact_overlap_ratio, 4),
        normalized_overlap_count=normalized_overlap,
        normalized_overlap_ratio=round(normalized_ratio, 4),
        jaccard=round(jaccard, 4),
        containment_source_to_target=round(containment_s_to_t, 4),
        containment_target_to_source=round(containment_t_to_s, 4),
        mode=mode,
        sample_size=sample_size,
        common_values=list(overlap)[:20],
        confidence_flags=flags,
    )


def compute_value_evidence_batch(
    table_profiles: Dict[str, TableProfile],
    candidates,
    sample_size: int = 100,
) -> Dict[Tuple[str, str, str, str], ValueEvidence]:
    """Compute value evidence for all candidates in batch.

    Args:
        table_profiles: Mapping of table key -> TableProfile.
        candidates: List of Candidate objects.
        sample_size: Sample size threshold.

    Returns:
        Mapping of (source_table, source_col, target_table, target_col) -> ValueEvidence.
    """
    results: Dict[Tuple[str, str, str, str], ValueEvidence] = {}

    for c in candidates:
        src_table = c.source_table
        src_col = c.source_column
        tgt_table = c.target_table
        tgt_col = c.target_column

        src_profile = table_profiles.get(src_table, {}).columns.get(src_col)
        tgt_profile = table_profiles.get(tgt_table, {}).columns.get(tgt_col)

        if src_profile and tgt_profile:
            evidence = compute_value_evidence(src_profile, tgt_profile, sample_size)
            results[(src_table, src_col, tgt_table, tgt_col)] = evidence
        else:
            results[(src_table, src_col, tgt_table, tgt_col)] = ValueEvidence(
                mode="missing", sample_size=sample_size
            )

    return results