"""Pipeline orchestrator.

Runs the full analysis pipeline: candidates -> type compat -> value evidence
-> string evidence -> scoring -> suggested edges.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.candidates import Candidate, generate_candidates
from src.models import ColumnProfile, ColumnInfo, TableMetadata, TableProfile
from src.scoring import ScoreBreakdown, compute_score
from src.string_evidence import StringEvidence, compute_string_evidence
from src.string_profiler import StringProfile
from src.type_compat import TypeDecision, evaluate_type_compatibility, filter_compatible_candidates
from src.types import canonicalize_type
from src.value_evidence import ValueEvidence, compute_value_evidence


@dataclass
class AnalysisResult:
    """Complete result of the analysis pipeline."""
    candidates: List[Candidate] = field(default_factory=list)
    type_decisions: Dict[Tuple[str, str, str, str], TypeDecision] = field(default_factory=dict)
    value_evidence: Dict[Tuple[str, str, str, str], ValueEvidence] = field(default_factory=dict)
    string_evidence: Dict[Tuple[str, str, str, str], StringEvidence] = field(default_factory=dict)
    scores: Dict[Tuple[str, str, str, str], ScoreBreakdown] = field(default_factory=dict)
    rejected_candidates: List[Candidate] = field(default_factory=list)

    @property
    def suggested_edges(self) -> List[Dict[str, Any]]:
        """Convert scores to suggested edge dicts for the graph."""
        edges = []
        for key, breakdown in self.scores.items():
            if breakdown.confidence_band == "reject":
                continue
            src_table, src_col, tgt_table, tgt_col = key
            # Find the candidate for this pair
            candidate = None
            for c in self.candidates:
                if (c.source_table == src_table and c.source_column == src_col
                        and c.target_table == tgt_table and c.target_column == tgt_col):
                    candidate = c
                    break

            evidence: Dict[str, Any] = {
                "name_score": breakdown.name_score,
                "type_score": breakdown.type_score,
                "value_score": breakdown.value_score,
                "uniqueness_score": breakdown.uniqueness_score,
                "null_score": breakdown.null_score,
                "match_type": candidate.match_type if candidate else "unknown",
                "reason": candidate.reason if candidate else "",
            }

            ve = self.value_evidence.get(key)
            if ve:
                evidence["value_evidence"] = {
                    "exact_overlap_count": ve.exact_overlap_count,
                    "exact_overlap_ratio": ve.exact_overlap_ratio,
                    "jaccard": ve.jaccard,
                    "mode": ve.mode,
                    "confidence_flags": ve.confidence_flags,
                }

            se = self.string_evidence.get(key)
            if se:
                evidence["string_evidence"] = {
                    "categorical_alignment": se.categorical_alignment,
                    "token_similarity": se.token_similarity,
                    "fuzzy_match_score": se.fuzzy_match_score,
                    "is_both_categorical": se.is_both_categorical,
                }

            edges.append({
                "source_table": src_table,
                "source_column": src_col,
                "target_table": tgt_table,
                "target_column": tgt_col,
                "confidence": breakdown.confidence,
                "confidence_band": breakdown.confidence_band,
                "evidence": evidence,
            })

        # Sort by confidence descending
        edges.sort(key=lambda e: e["confidence"], reverse=True)
        return edges


def run_analysis(
    tables: Dict[str, TableMetadata],
    table_profiles: Dict[str, TableProfile],
    string_profiles: Optional[Dict[str, Dict[str, StringProfile]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """Run the full relationship analysis pipeline.

    Steps:
    1. Generate candidates (name similarity + alias + type prefilter)
    2. Evaluate type compatibility
    3. Compute value evidence (overlap, Jaccard, containment)
    4. Compute string evidence (categorical alignment, token similarity)
    5. Score candidates with weighted formula
    6. Filter by confidence band

    Args:
        tables: Dictionary of table metadata keyed by `{schema}.{table}`.
        table_profiles: Dictionary of table profiles keyed by `{schema}.{table}`.
        string_profiles: Optional string profiles keyed by `{schema}.{table}` -> `{col_name}` -> StringProfile.
        config: Configuration dict with thresholds, aliases, exclusions.

    Returns:
        AnalysisResult with all evidence and scores.
    """
    if config is None:
        config = {}

    thresholds = config.get("thresholds", {})
    profiling_config = config.get("profiling", {})
    sample_size = profiling_config.get("sample_size", 100)

    result = AnalysisResult()

    # Step 1: Generate candidates
    candidates = generate_candidates(tables, config)
    result.candidates = candidates

    if not candidates:
        return result

    # Step 2: Type compatibility
    compatible, rejected = filter_compatible_candidates(candidates, config)
    result.rejected_candidates = rejected

    # Store type decisions
    for c in candidates:
        key = (c.source_table, c.source_column, c.target_table, c.target_column)
        if hasattr(c, "type_decision"):
            result.type_decisions[key] = c.type_decision  # type: ignore[attr-defined]

    if not compatible:
        return result

    # Build profile lookups
    profile_lookup: Dict[Tuple[str, str], ColumnProfile] = {}
    for table_key, t_profile in table_profiles.items():
        for col_name, col_profile in t_profile.columns.items():
            profile_lookup[(table_key, col_name)] = col_profile

    # Step 3: Value evidence
    for c in compatible:
        key = (c.source_table, c.source_column, c.target_table, c.target_column)
        src_prof = profile_lookup.get((c.source_table, c.source_column))
        tgt_prof = profile_lookup.get((c.target_table, c.target_column))

        if src_prof and tgt_prof:
            ve = compute_value_evidence(src_prof, tgt_prof, sample_size)
            result.value_evidence[key] = ve
        else:
            result.value_evidence[key] = ValueEvidence(mode="missing")

    # Step 4: String evidence (if available)
    if string_profiles:
        for c in compatible:
            key = (c.source_table, c.source_column, c.target_table, c.target_column)
            sp_a = string_profiles.get(c.source_table, {}).get(c.source_column)
            sp_b = string_profiles.get(c.target_table, {}).get(c.target_column)

            if sp_a and sp_b:
                se = compute_string_evidence(sp_a, sp_b)
                result.string_evidence[key] = se

    # Step 5: Scoring
    for c in compatible:
        key = (c.source_table, c.source_column, c.target_table, c.target_column)
        decision = result.type_decisions.get(key)
        if decision is None:
            continue

        ve = result.value_evidence.get(key)
        v_score = ve.value_score if ve else 0.0

        src_prof = profile_lookup.get((c.source_table, c.source_column))
        tgt_prof = profile_lookup.get((c.target_table, c.target_column))

        breakdown = compute_score(
            name_score=c.name_score,
            type_decision=decision,
            value_score=v_score,
            source_profile=src_prof,
            target_profile=tgt_prof,
            weights=None,
        )
        result.scores[key] = breakdown

    return result