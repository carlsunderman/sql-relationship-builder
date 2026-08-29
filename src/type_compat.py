"""Type compatibility evaluator.

Extends the basic type canonicalization with a detailed compatibility
decision including risk tags (None, cast_required, incompatible).
"""

from dataclasses import dataclass
from typing import Dict, Optional

from src.types import canonicalize_type, CanonicalType


@dataclass
class TypeDecision:
    """Result of type compatibility evaluation."""
    compatible: bool
    canonical_a: str
    canonical_b: str
    risk: Optional[str]  # None = safe, "cast_required", "incompatible"
    reason: str = ""


# Relaxed compatibility: allowed cross-family pairs
_RELAXED_PAIRS = frozenset({
    frozenset({CanonicalType.NUMERIC, CanonicalType.STRING}),
})


def evaluate_type_compatibility(
    type_a: str,
    type_b: str,
    strict: bool = True,
) -> TypeDecision:
    """Evaluate whether two SQL Server types are compatible for joining.

    Args:
        type_a: Native type string from column A.
        type_b: Native type string from column B.
        strict: If True, only same canonical family is compatible.
                If False, also allow NUMERIC <-> STRING with cast_required risk.

    Returns:
        TypeDecision with compatibility verdict and risk.
    """
    canon_a = canonicalize_type(type_a)
    canon_b = canonicalize_type(type_b)

    if canon_a == canon_b:
        return TypeDecision(
            compatible=True,
            canonical_a=canon_a.value,
            canonical_b=canon_b.value,
            risk=None,
            reason=f"Same canonical family: {canon_a.value}",
        )

    if not strict and frozenset({canon_a, canon_b}) in _RELAXED_PAIRS:
        return TypeDecision(
            compatible=True,
            canonical_a=canon_a.value,
            canonical_b=canon_b.value,
            risk="cast_required",
            reason=f"Cross-family cast: {canon_a.value} <-> {canon_b.value}",
        )

    return TypeDecision(
        compatible=False,
        canonical_a=canon_a.value,
        canonical_b=canon_b.value,
        risk="incompatible",
        reason=f"Incompatible families: {canon_a.value} vs {canon_b.value}",
    )


def filter_compatible_candidates(
    candidates,
    config: Optional[Dict] = None,
):
    """Filter and annotate candidates with type compatibility decisions.

    Args:
        candidates: List of Candidate objects from generate_candidates().
        config: Configuration dict with type_compatibility setting.

    Returns:
        Tuple of (compatible_candidates, rejected_candidates).
    """
    if config is None:
        config = {}

    thresholds = config.get("thresholds", {})
    compat_mode = thresholds.get("type_compatibility", "strict")
    strict = compat_mode == "strict"

    compatible = []
    rejected = []

    for c in candidates:
        decision = evaluate_type_compatibility(c.source_type, c.target_type, strict=strict)
        # Attach decision to the candidate (extend dynamically)
        c.type_decision = decision  # type: ignore[attr-defined]

        if decision.compatible:
            compatible.append(c)
        else:
            rejected.append(c)

    return compatible, rejected