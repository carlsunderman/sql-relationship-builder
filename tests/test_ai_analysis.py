"""Tests for AI analysis batching/token guardrails."""

from typing import Any, cast

from src.ai_analysis import (
    _build_user_message_with_guardrail,
    _call_llm_with_retries,
    _format_exception_with_cause,
    _is_retryable_error,
    _parse_json_array_response,
    discover_relationships_batched,
    generate_column_descriptions_batched,
)
from src.llm import LLMConfig
from src.models import ColumnInfo, TableMetadata


def _make_table(table_name: str, col_count: int, name_len: int) -> TableMetadata:
    cols = []
    for i in range(col_count):
        col_name = f"col_{i}_" + ("x" * name_len)
        cols.append(
            ColumnInfo(
                name=col_name,
                data_type="varchar",
                is_nullable=True,
                ordinal_position=i + 1,
            )
        )

    return TableMetadata(
        schema_name="dbo",
        table_name=table_name,
        row_count=1000,
        columns=cols,
    )


def test_guardrail_requests_batch_split_when_payload_too_large() -> None:
    tables = {
        "dbo.t1": _make_table("t1", col_count=20, name_len=1200),
        "dbo.t2": _make_table("t2", col_count=20, name_len=1200),
        "dbo.t3": _make_table("t3", col_count=20, name_len=1200),
        "dbo.t4": _make_table("t4", col_count=20, name_len=1200),
    }

    user_msg, split_batches, notes = _build_user_message_with_guardrail(
        batch_keys=["dbo.t1", "dbo.t2", "dbo.t3", "dbo.t4"],
        tables=tables,
        profiles={},
        max_columns=50,
        prefix="Describe each column in these tables.\n\n",
    )

    assert user_msg == ""
    assert len(split_batches) == 2
    assert split_batches[0] == ["dbo.t1", "dbo.t2"]
    assert split_batches[1] == ["dbo.t3", "dbo.t4"]
    assert isinstance(notes, list)


def test_guardrail_truncates_single_large_table_payload() -> None:
    # name_len must be large enough that even _MIN_COLUMNS_PER_TABLE (8) columns
    # exceed _MAX_BATCH_PROMPT_CHARS (35 000): ceil(35000/8) ≈ 4375.
    tables = {
        "dbo.wide": _make_table("wide", col_count=20, name_len=4400),
    }

    user_msg, split_batches, notes = _build_user_message_with_guardrail(
        batch_keys=["dbo.wide"],
        tables=tables,
        profiles={},
        max_columns=50,
        prefix="Find relationships.\n\n",
    )

    assert split_batches == []
    assert "[Truncated due to prompt size limits" in user_msg
    assert any("truncated" in n.lower() for n in notes)


class _FailingClient:
    def __init__(self, errors: list[Exception], final_response: str = "[]") -> None:
        self._errors = errors
        self._final_response = final_response
        self.calls = 0
        self.config = LLMConfig(model="x", endpoint="http://localhost", api_key="k")

    def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._final_response


def test_retryable_error_detection_and_cause_formatting() -> None:
    root = RuntimeError("timed out while reading")
    err = RuntimeError("Connection error.")
    err.__cause__ = root

    msg = _format_exception_with_cause(err)
    assert "Connection error." in msg
    assert "timed out while reading" in msg
    assert _is_retryable_error(err)


def test_retry_detector_does_not_treat_char_position_as_http_code() -> None:
    err = RuntimeError("Expecting ',' delimiter: line 17 column 237 (char 3429)")
    assert not _is_retryable_error(err)


def test_parse_json_array_response_handles_wrapper_and_trailing_comma() -> None:
    wrapped = '{"results": [{"table": "dbo.t", "column": "c",},]}'
    parsed = _parse_json_array_response(wrapped)
    assert len(parsed) == 1
    assert parsed[0]["table"] == "dbo.t"


def test_call_llm_with_retries_succeeds_after_transient_failure() -> None:
    client = _FailingClient([RuntimeError("Connection error.")], final_response="[]")

    result = _call_llm_with_retries(
        cast(Any, client),
        messages=[{"role": "user", "content": "x"}],
        max_tokens=100,
        timeout=30,
        retries=2,
        retry_delay_seconds=0.0,
    )

    assert result == "[]"
    assert client.calls == 2


def test_generate_column_descriptions_splits_batch_after_retryable_error() -> None:
    tables = {
        "dbo.t1": _make_table("t1", col_count=5, name_len=3),
        "dbo.t2": _make_table("t2", col_count=5, name_len=3),
    }
    # _call_llm_with_retries makes 3 attempts (retries=2); need 3 errors to exhaust them.
    client = _FailingClient(
        [RuntimeError("Connection error.")] * 3,
        final_response="[]",
    )

    _, errors = generate_column_descriptions_batched(
        cast(Any, client),
        tables,
        profiles={},
        domain_context="",
        batch_size=2,
        timeout=10,
    )

    assert any("retrying in smaller batches" in e.lower() for e in errors)


def test_discover_relationships_splits_batch_after_retryable_error() -> None:
    tables = {
        "dbo.t1": _make_table("t1", col_count=5, name_len=3),
        "dbo.t2": _make_table("t2", col_count=5, name_len=3),
    }
    # _call_llm_with_retries makes 3 attempts (retries=2); need 3 errors to exhaust them.
    client = _FailingClient(
        [RuntimeError("Connection error.")] * 3,
        final_response="[]",
    )

    _, errors = discover_relationships_batched(
        cast(Any, client),
        tables,
        profiles={},
        domain_context="",
        existing_relationships=[],
        batch_size=2,
        timeout=10,
    )

    assert any("retrying in smaller batches" in e.lower() for e in errors)
