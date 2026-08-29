"""Priority-aware relationship candidate generator.

Generates candidate column pairs across tables using:
1. Name similarity (Levenshtein-based ratio)
2. Alias dictionary (config-driven synonyms)
3. Type prefilter (same canonical family)

Candidates are prioritized: exact name > alias > fuzzy name.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models import ColumnInfo, TableMetadata
from src.types import canonicalize_type, CanonicalType


@dataclass
class Candidate:
    """A candidate relationship between two columns."""
    source_table: str
    source_column: str
    source_type: str
    target_table: str
    target_column: str
    target_type: str
    match_type: str = "fuzzy"
    name_score: float = 0.0
    reason: str = ""


def _normalize_name(name: str) -> str:
    """Normalize a column name for comparison.

    Lowercase, strip common suffixes like '_id', '_key', '_code'.
    """
    n = name.lower().strip()
    # Strip common key suffixes
    for suffix in ("_id", "_key", "_code", "_no", "_num", "_number"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n


def _name_similarity(name_a: str, name_b: str) -> float:
    """Compute a normalized similarity score between two column names.

    Uses SequenceMatcher ratio on the raw names, then boosts if
    the normalized (suffix-stripped) names match.
    """
    raw = SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()

    norm_a = _normalize_name(name_a)
    norm_b = _normalize_name(name_b)

    if norm_a and norm_b and norm_a == norm_b:
        return max(raw, 0.9)

    return raw


def _build_alias_index(aliases: Dict[str, List[str]]) -> Dict[str, str]:
    """Build a reverse lookup: column_name -> canonical_alias_group.

    Args:
        aliases: Config aliases dict mapping group -> [names].

    Returns:
        Mapping of lowercase column name to canonical group name.
    """
    index: Dict[str, str] = {}
    for group, names in aliases.items():
        for name in names:
            index[name.lower().strip()] = group
    return index


def generate_candidates(
    tables: Dict[str, TableMetadata],
    config: Optional[Dict[str, Any]] = None,
) -> List[Candidate]:
    """Generate relationship candidates across all table pairs.

    Strategy:
    1. Exact name match + compatible type -> highest priority
    2. Alias match + compatible type -> high priority
    3. Fuzzy name similarity + compatible type -> lower priority

    Args:
        tables: Dictionary of table metadata keyed by `{schema}.{table}`.
        config: Configuration dict with thresholds, aliases, exclusions.

    Returns:
        List of Candidate objects, sorted by priority (name_score desc).
    """
    if config is None:
        config = {}

    thresholds = config.get("thresholds", {})
    sim_min = thresholds.get("name_similarity_min", 0.6)
    aliases = config.get("aliases", {})

    exclusions = config.get("exclusions", {})
    excluded_tables = set(exclusions.get("table_patterns", []))
    excluded_columns = set(exclusions.get("column_patterns", []))

    alias_index = _build_alias_index(aliases)

    # Build column lookup per table
    table_columns: Dict[str, Dict[str, ColumnInfo]] = {}
    for key, meta in tables.items():
        cols: Dict[str, ColumnInfo] = {}
        for col in meta.columns:
            # Skip excluded columns
            skip = False
            for pattern in excluded_columns:
                if pattern.lower() in col.name.lower():
                    skip = True
                    break
            if not skip:
                cols[col.name] = col
        table_columns[key] = cols

    # Filter excluded tables
    filtered_tables = {}
    for key, meta in tables.items():
        skip = False
        for pattern in excluded_tables:
            if pattern.lower() in key.lower():
                skip = True
                break
        if not skip:
            filtered_tables[key] = meta

    candidates: List[Candidate] = []
    seen_pairs: Set[Tuple[str, str, str, str]] = set()

    table_keys = sorted(filtered_tables.keys())

    for i, key_a in enumerate(table_keys):
        meta_a = filtered_tables[key_a]
        cols_a = table_columns.get(key_a, {})
        for j in range(i + 1, len(table_keys)):
            key_b = table_keys[j]
            meta_b = filtered_tables[key_b]
            cols_b = table_columns.get(key_b, {})

            # Strategy 1: Exact name match
            common_names = set(cols_a.keys()) & set(cols_b.keys())
            for col_name in common_names:
                ca = cols_a[col_name]
                cb = cols_b[col_name]
                if canonicalize_type(ca.data_type) == canonicalize_type(cb.data_type):
                    pair = (key_a, col_name, key_b, col_name)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        candidates.append(Candidate(
                            source_table=key_a,
                            source_column=col_name,
                            source_type=ca.data_type,
                            target_table=key_b,
                            target_column=col_name,
                            target_type=cb.data_type,
                            match_type="exact_name",
                            name_score=1.0,
                            reason=f"Exact name match: {col_name}",
                        ))

            # Strategy 2: Alias match
            for col_a_name, ca in cols_a.items():
                group_a = alias_index.get(col_a_name.lower())
                if group_a is None:
                    continue
                for col_b_name, cb in cols_b.items():
                    group_b = alias_index.get(col_b_name.lower())
                    if group_a != group_b:
                        continue
                    if canonicalize_type(ca.data_type) == canonicalize_type(cb.data_type):
                        pair = (key_a, col_a_name, key_b, col_b_name)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            candidates.append(Candidate(
                                source_table=key_a,
                                source_column=col_a_name,
                                source_type=ca.data_type,
                                target_table=key_b,
                                target_column=col_b_name,
                                target_type=cb.data_type,
                                match_type="alias",
                                name_score=0.85,
                                reason=f"Alias group '{group_a}': {col_a_name} <-> {col_b_name}",
                            ))

            # Strategy 3: Fuzzy name similarity
            for col_a_name, ca in cols_a.items():
                for col_b_name, cb in cols_b.items():
                    pair = (key_a, col_a_name, key_b, col_b_name)
                    if pair in seen_pairs:
                        continue
                    sim = _name_similarity(col_a_name, col_b_name)
                    if sim >= sim_min:
                        if canonicalize_type(ca.data_type) == canonicalize_type(cb.data_type):
                            seen_pairs.add(pair)
                            candidates.append(Candidate(
                                source_table=key_a,
                                source_column=col_a_name,
                                source_type=ca.data_type,
                                target_table=key_b,
                                target_column=col_b_name,
                                target_type=cb.data_type,
                                match_type="fuzzy",
                                name_score=round(sim, 4),
                                reason=f"Name similarity {sim:.2f}: {col_a_name} <-> {col_b_name}",
                            ))

    # Sort by name_score descending
    candidates.sort(key=lambda c: c.name_score, reverse=True)
    return candidates