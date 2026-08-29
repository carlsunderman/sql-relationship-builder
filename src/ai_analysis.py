"""AI-powered relationship analysis.

Uses an LLM to:
1. Generate descriptive text for columns based on domain context
2. Discover relationships that deterministic analysis might miss
   (e.g., knowing 'Block' and 'blk' refer to the same concept in Oil & Gas)

Handles large schemas by batching tables and compacting context.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.graph import SuggestedEdge
from src.llm import LLMClient
from src.models import TableMetadata, TableProfile


@dataclass
class AIColumnDescription:
    """AI-generated description for a column."""

    table: str
    column: str
    description: str
    data_type: str = ""
    domain_terms: List[str] = field(default_factory=list)


@dataclass
class AIRelationship:
    """AI-suggested relationship between two columns."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    confidence: float = 0.0
    reasoning: str = ""
    rel_type: str = "one-to-many"


@dataclass
class AIAnalysisResult:
    """Complete result of AI analysis."""

    column_descriptions: List[AIColumnDescription] = field(default_factory=list)
    relationships: List[AIRelationship] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_table_context(
    table_key: str,
    metadata: TableMetadata,
    profile: Optional[TableProfile],
    max_columns: int = 50,
) -> str:
    """Build a compact text description of a table for the LLM.

    Only sends column names and types, plus lightweight profile stats
    (null ratio, distinct ratio, uniqueness). No top values or min/max
    to keep the payload small.
    """
    lines = [f"Table: {table_key} (rows: {metadata.row_count:,})"]

    for i, col in enumerate(metadata.columns):
        if i >= max_columns:
            lines.append(
                f"  ... and {len(metadata.columns) - max_columns} more columns"
            )
            break

        parts = [f"  {col.name} ({col.data_type}, nullable={col.is_nullable})"]
        if profile and col.name in profile.columns:
            cp = profile.columns[col.name]
            parts.append(
                f"    null_ratio={cp.null_ratio:.2f} "
                f"distinct={cp.distinct_count} "
                f"uniqueness={cp.uniqueness_ratio:.3f}"
            )
        lines.append(" ".join(parts))

    return "\n".join(lines)


def _build_system_prompt_descriptions(domain_context: str) -> str:
    prompt = "You are a database relationship expert. Generate concise, accurate descriptions for database columns."
    if domain_context.strip():
        prompt += (
            f"\n\nDomain context:\n{domain_context}\n"
            "Use this context to make descriptions specific and accurate."
        )
    prompt += (
        "\n\nFor each column provide a JSON object:\n"
        '{"table": "schema.table", "column": "col", "description": "...", "data_type": "...", "domain_terms": ["t1"]}\n\n'
        "Respond with a JSON array only. No markdown fences, no extra text."
    )
    return prompt


def _build_system_prompt_relationships(domain_context: str) -> str:
    prompt = (
        "You are a database relationship expert. Identify relationships between database tables "
        "that may not be obvious from column names alone. You understand domain-specific "
        "naming conventions (e.g. 'Block' and 'blk' both mean drilling blocks in Oil & Gas)."
    )
    if domain_context.strip():
        prompt += (
            f"\n\nDomain context:\n{domain_context}\n"
            "Use this context to find domain-specific relationships."
        )
    prompt += (
        "\n\nFor each relationship provide a JSON object:\n"
        '{"source_table": "s.t", "source_column": "c", "target_table": "s.t", '
        '"target_column": "c", "confidence": 0.85, "reasoning": "...", "rel_type": "one-to-many"}\n\n'
        "Respond with a JSON array only. No markdown fences, no extra text."
    )
    return prompt


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract the JSON array from LLM output."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _looks_truncated(response: str) -> bool:
    """Return True when the response appears cut off before the closing ] of the JSON array."""
    return not _extract_json(response).strip().endswith("]")


def _format_exception_with_cause(exc: Exception) -> str:
    """Include nested cause/context details to avoid opaque 'Connection error.' messages."""
    parts: List[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc

    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = str(cur).strip() or cur.__class__.__name__
        if text not in parts:
            parts.append(text)
        cur = cur.__cause__ or cur.__context__

    return " | Caused by: ".join(parts)


def _is_retryable_error(exc: Exception) -> bool:
    """Best-effort detector for transient network/service failures."""
    msg = _format_exception_with_cause(exc).lower()
    retry_tokens = [
        "connection error",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "server disconnected",
        "connection reset",
        "rate limit",
        "too many requests",
    ]
    if any(token in msg for token in retry_tokens):
        return True

    # Match HTTP status codes as whole tokens (avoid false positives like "char 3429").
    return re.search(r"\b(429|502|503|504)\b", msg) is not None


def _call_llm_with_retries(
    client: LLMClient,
    messages: List[Dict[str, str]],
    max_tokens: int,
    timeout: int,
    retries: int = 2,
    retry_delay_seconds: float = 1.0,
) -> str:
    """Call the LLM with simple retry/backoff for transient errors."""
    attempts = max(1, retries + 1)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            return client.chat(
                messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - keep broad to preserve root cause text
            last_error = exc
            if attempt >= attempts or not _is_retryable_error(exc):
                break
            time.sleep(retry_delay_seconds * attempt)

    assert last_error is not None
    details = _format_exception_with_cause(last_error)
    raise RuntimeError(f"{details} (after {attempts} attempt(s))")


def _parse_json_array_response(text: str) -> List[Dict[str, Any]]:
    """Parse model output into a list of dicts with light cleanup and wrapper support."""

    def _coerce(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for k in ("results", "items", "data", "relationships", "descriptions"):
                v = payload.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
        raise ValueError("Expected JSON array (or wrapper object with array field).")

    base = _extract_json(text).strip()

    # Attempt 1: direct parse
    try:
        return _coerce(json.loads(base))
    except Exception as e1:
        # Attempt 2: common cleanup (smart quotes, trailing commas)
        cleaned = (
            base.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            return _coerce(json.loads(cleaned))
        except Exception as e2:
            snippet = base[:260].replace("\n", " ")
            raise ValueError(
                f"Invalid JSON response. First error: {e1}. After cleanup: {e2}. "
                f"Response snippet: {snippet}"
            ) from e2


def _repair_json_array_with_llm(
    client: LLMClient,
    broken_text: str,
    expected_item_shape: str,
    timeout: int,
) -> str:
    """Ask the LLM to repair malformed JSON into a strict JSON array."""
    repair_system = (
        "You fix malformed JSON. Return valid JSON only. No markdown, no explanations."
    )
    repair_user = (
        "The following text is intended to be a JSON array but is malformed. "
        f"Repair it into a valid JSON array of objects shaped like: {expected_item_shape}.\n\n"
        "Malformed text:\n"
        f"{broken_text}"
    )

    return _call_llm_with_retries(
        client,
        [
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ],
        max_tokens=min(max(client.config.max_tokens, 512), 4096),
        timeout=timeout,
        retries=1,
    )


# ---------------------------------------------------------------------------
# Batched analysis
# ---------------------------------------------------------------------------

# Approximate prompt-size guardrail to reduce context-window errors on large schemas.
# (rough char budget, model-agnostic)
_MAX_BATCH_PROMPT_CHARS = 35000
_MIN_COLUMNS_PER_TABLE = 8


def _batch_tables(tables: Dict[str, TableMetadata], batch_size: int) -> List[List[str]]:
    """Split table keys into batches."""
    keys = list(tables.keys())
    return [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]


def _build_user_message_with_guardrail(
    batch_keys: List[str],
    tables: Dict[str, TableMetadata],
    profiles: Dict[str, TableProfile],
    max_columns: int,
    prefix: str,
) -> tuple[str, List[List[str]], List[str]]:
    """Build a batch user message with adaptive shrink/split for large payloads.

    Returns:
        (user_message, split_batches, notes)
        - user_message: ready-to-send message, or "" when split is requested.
        - split_batches: two sub-batches when current batch should be split.
        - notes: informational warnings for truncation/shrinking.
    """
    notes: List[str] = []
    curr_max_columns = max(1, int(max_columns))

    while True:
        blocks = [
            _build_table_context(k, tables[k], profiles.get(k), curr_max_columns)
            for k in batch_keys
        ]
        user_msg = prefix + "\n\n".join(blocks)

        if len(user_msg) <= _MAX_BATCH_PROMPT_CHARS:
            if curr_max_columns < max_columns:
                notes.append(
                    "Prompt was large; reduced columns-per-table "
                    f"from {max_columns} to {curr_max_columns} for this batch."
                )
            return user_msg, [], notes

        if curr_max_columns > _MIN_COLUMNS_PER_TABLE:
            next_cols = max(_MIN_COLUMNS_PER_TABLE, curr_max_columns // 2)
            if next_cols == curr_max_columns:
                next_cols = max(_MIN_COLUMNS_PER_TABLE, curr_max_columns - 1)
            curr_max_columns = next_cols
            continue

        if len(batch_keys) > 1:
            mid = len(batch_keys) // 2
            return "", [batch_keys[:mid], batch_keys[mid:]], notes

        # Single very-wide table: hard truncate to stay under payload budget.
        truncated = user_msg[:_MAX_BATCH_PROMPT_CHARS]
        truncated += (
            "\n\n[Truncated due to prompt size limits; consider lowering batch size.]"
        )
        notes.append(
            "Single table context exceeded prompt budget and was truncated. "
            "Consider reducing columns profiled or using a model with larger context window."
        )
        return truncated, [], notes


def generate_column_descriptions_batched(
    client: LLMClient,
    tables: Dict[str, TableMetadata],
    profiles: Dict[str, TableProfile],
    domain_context: str,
    batch_size: int = 8,
    max_columns: int = 50,
    timeout: int = 120,
) -> tuple[List[AIColumnDescription], List[str]]:
    """Generate descriptions in batches to avoid token/timeout limits."""
    system_prompt = _build_system_prompt_descriptions(domain_context)
    pending_batches = _batch_tables(tables, batch_size)
    all_results: List[AIColumnDescription] = []
    errors: List[str] = []

    while pending_batches:
        batch_keys = pending_batches.pop(0)

        user_msg, split_batches, notes = _build_user_message_with_guardrail(
            batch_keys=batch_keys,
            tables=tables,
            profiles=profiles,
            max_columns=max_columns,
            prefix="Describe each column in these tables.\n\n",
        )

        if split_batches:
            pending_batches = split_batches + pending_batches
            continue

        errors.extend(
            [
                f"Column descriptions batch ({len(batch_keys)} tables): {n}"
                for n in notes
            ]
        )

        try:
            response = _call_llm_with_retries(
                client,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=client.config.max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            if len(batch_keys) > 1 and _is_retryable_error(e):
                mid = len(batch_keys) // 2
                pending_batches = [batch_keys[:mid], batch_keys[mid:]] + pending_batches
                errors.append(
                    "Column descriptions batch "
                    f"({len(batch_keys)} tables): transient failure, retrying in smaller batches. "
                    f"Details: {_format_exception_with_cause(e)}"
                )
                continue
            errors.append(
                "Column descriptions batch "
                f"({len(batch_keys)} tables): {_format_exception_with_cause(e)}"
            )
            continue

        try:
            data = _parse_json_array_response(response)
        except Exception as parse_err:
            if len(batch_keys) > 1:
                mid = len(batch_keys) // 2
                pending_batches = [batch_keys[:mid], batch_keys[mid:]] + pending_batches
                errors.append(
                    "Column descriptions batch "
                    f"({len(batch_keys)} tables): response malformed or truncated, "
                    "retrying in smaller batches."
                )
                continue
            # Single-table: attempt repair once; if it fails, report with truncation hint.
            try:
                repaired = _repair_json_array_with_llm(
                    client,
                    response,
                    '{"table": "schema.table", "column": "name", "description": "text", "data_type": "type", "domain_terms": ["term"]}',
                    timeout=timeout,
                )
                data = _parse_json_array_response(repaired)
                errors.append(
                    "Column descriptions batch "
                    f"({len(batch_keys)} tables): model returned malformed JSON; auto-repaired. "
                    f"Original parse error: {parse_err}"
                )
            except Exception:
                hint = (
                    " Response appears truncated — increase Max Tokens and retry."
                    if _looks_truncated(response) else ""
                )
                errors.append(
                    "Column descriptions batch "
                    f"({len(batch_keys)} tables): {parse_err}.{hint}"
                )
                continue

        for item in data:
            all_results.append(
                AIColumnDescription(
                    table=item.get("table", ""),
                    column=item.get("column", ""),
                    description=item.get("description", ""),
                    data_type=item.get("data_type", ""),
                    domain_terms=item.get("domain_terms", []),
                )
            )

    return all_results, errors


def discover_relationships_batched(
    client: LLMClient,
    tables: Dict[str, TableMetadata],
    profiles: Dict[str, TableProfile],
    domain_context: str,
    existing_relationships: Optional[List[Dict[str, str]]] = None,
    batch_size: int = 8,
    max_columns: int = 50,
    timeout: int = 120,
) -> tuple[List[AIRelationship], List[str]]:
    """Discover relationships in batches to handle large schemas."""
    system_prompt = _build_system_prompt_relationships(domain_context)

    existing_str = ""
    if existing_relationships:
        lines = [
            f"  {r.get('source', '?')}.{r.get('source_col', '?')} -> "
            f"{r.get('target', '?')}.{r.get('target_col', '?')}"
            for r in existing_relationships[:50]  # cap to avoid bloating prompt
        ]
        existing_str = "\nKnown relationships (skip these):\n" + "\n".join(lines)

    pending_batches = _batch_tables(tables, batch_size)
    all_results: List[AIRelationship] = []
    errors: List[str] = []

    while pending_batches:
        batch_keys = pending_batches.pop(0)

        user_msg, split_batches, notes = _build_user_message_with_guardrail(
            batch_keys=batch_keys,
            tables=tables,
            profiles=profiles,
            max_columns=max_columns,
            prefix=(
                "Find non-obvious relationships between these tables."
                f"{existing_str}\n\n"
            ),
        )

        if split_batches:
            pending_batches = split_batches + pending_batches
            continue

        errors.extend(
            [f"Relationship batch ({len(batch_keys)} tables): {n}" for n in notes]
        )

        try:
            response = _call_llm_with_retries(
                client,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=client.config.max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            if len(batch_keys) > 1 and _is_retryable_error(e):
                mid = len(batch_keys) // 2
                pending_batches = [batch_keys[:mid], batch_keys[mid:]] + pending_batches
                errors.append(
                    "Relationship batch "
                    f"({len(batch_keys)} tables): transient failure, retrying in smaller batches. "
                    f"Details: {_format_exception_with_cause(e)}"
                )
                continue
            errors.append(
                f"Relationship batch ({len(batch_keys)} tables): {_format_exception_with_cause(e)}"
            )
            continue

        try:
            data = _parse_json_array_response(response)
        except Exception as parse_err:
            if len(batch_keys) > 1:
                mid = len(batch_keys) // 2
                pending_batches = [batch_keys[:mid], batch_keys[mid:]] + pending_batches
                errors.append(
                    "Relationship batch "
                    f"({len(batch_keys)} tables): response malformed or truncated, "
                    "retrying in smaller batches."
                )
                continue
            try:
                repaired = _repair_json_array_with_llm(
                    client,
                    response,
                    '{"source_table": "s.t", "source_column": "c", "target_table": "s.t", "target_column": "c", "confidence": 0.85, "reasoning": "text", "rel_type": "one-to-many"}',
                    timeout=timeout,
                )
                data = _parse_json_array_response(repaired)
                errors.append(
                    "Relationship batch "
                    f"({len(batch_keys)} tables): model returned malformed JSON; auto-repaired. "
                    f"Original parse error: {parse_err}"
                )
            except Exception:
                hint = (
                    " Response appears truncated — increase Max Tokens and retry."
                    if _looks_truncated(response) else ""
                )
                errors.append(
                    f"Relationship batch ({len(batch_keys)} tables): {parse_err}.{hint}"
                )
                continue

        for item in data:
            all_results.append(
                AIRelationship(
                    source_table=item.get("source_table", ""),
                    source_column=item.get("source_column", ""),
                    target_table=item.get("target_table", ""),
                    target_column=item.get("target_column", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=item.get("reasoning", ""),
                    rel_type=item.get("rel_type", "one-to-many"),
                )
            )

    return all_results, errors


def run_ai_analysis(
    client: LLMClient,
    tables: Dict[str, TableMetadata],
    profiles: Dict[str, TableProfile],
    domain_context: str,
    existing_relationships: Optional[List[Dict[str, str]]] = None,
    batch_size: int = 8,
    max_columns: int = 50,
    timeout: int = 120,
) -> AIAnalysisResult:
    """Run full AI analysis with batching for large schemas.

    Args:
        client: Configured LLM client.
        tables: Table metadata keyed by {schema}.{table}.
        profiles: Table profiles keyed by same format.
        domain_context: User-provided domain description.
        existing_relationships: Known relationships to avoid duplicates.
        batch_size: Tables per LLM call (lower = safer for small context windows).
        max_columns: Max columns to send per table (truncates with "...").
        timeout: Seconds per LLM call before giving up.

    Returns:
        AIAnalysisResult with descriptions, relationships, and any partial errors.
    """
    result = AIAnalysisResult()

    descs, desc_errors = generate_column_descriptions_batched(
        client,
        tables,
        profiles,
        domain_context,
        batch_size,
        max_columns,
        timeout,
    )
    result.column_descriptions = descs
    result.errors.extend(desc_errors)

    rels, rel_errors = discover_relationships_batched(
        client,
        tables,
        profiles,
        domain_context,
        existing_relationships,
        batch_size,
        max_columns,
        timeout,
    )
    result.relationships = rels
    result.errors.extend(rel_errors)

    return result


def ai_relationships_to_suggested_edges(
    ai_result: AIAnalysisResult,
    node_lookup: Dict[str, str],
) -> List[SuggestedEdge]:
    """Convert AI relationships to SuggestedEdge objects for the graph."""
    edges = []
    for rel in ai_result.relationships:
        src_key = node_lookup.get(rel.source_table)
        tgt_key = node_lookup.get(rel.target_table)
        if not (src_key and tgt_key):
            continue

        if rel.confidence >= 0.85:
            band = "high"
        elif rel.confidence >= 0.70:
            band = "medium"
        else:
            band = "low"

        edges.append(
            SuggestedEdge(
                source_key=src_key,
                source_column=rel.source_column,
                target_key=tgt_key,
                target_column=rel.target_column,
                confidence=rel.confidence,
                confidence_band=band,
                evidence={
                    "source": "ai",
                    "reasoning": rel.reasoning,
                    "rel_type": rel.rel_type,
                },
            )
        )
    return edges
