"""Answer Synthesizer agent: question + results -> natural-language answer."""

import logging
from typing import Any, Dict, List, Optional

from src.chat.models import QueryResult, QueryStatus
from src.llm import LLMClient

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """Synthesizes a natural-language answer from a question and its results."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def synthesize(
        self,
        question: str,
        query_result: QueryResult,
        generated_sql: str = "",
    ) -> str:
        """Generate a natural-language answer from query results.

        Args:
            question: The original user question.
            query_result: The result of executing the generated SQL.
            generated_sql: The SQL that was executed (for transparency).

        Returns:
            A natural-language answer string.
        """
        if query_result.status == QueryStatus.BLOCKED:
            return (
                f"I can't run that query — it was blocked for safety: "
                f"{query_result.error_message}"
            )

        if query_result.status == QueryStatus.ERROR:
            return (
                f"The query failed with this error: {query_result.error_message}\n\n"
                f"**Generated SQL:**\n```sql\n{generated_sql}\n```\n\n"
                "Try rephrasing your question, or check that the referenced tables exist."
            )

        if not query_result.rows and not query_result.columns:
            return "The query returned no results."

        system_prompt = (
            "You are a data analyst. Given a user's question, the SQL you ran, "
            "and the results, write a clear, concise natural-language answer. "
            "Reference specific values from the results. If the result set is large, "
            "summarize key findings rather than listing every row."
        )

        user_content = self._format_results(question, query_result, generated_sql)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            return self.llm.chat(messages, temperature=0.2)
        except Exception as e:
            logger.error("Answer synthesis failed: %s", e)
            # Fall back to a plain tabular summary
            return self._fallback_summary(query_result)

    def _format_results(
        self,
        question: str,
        result: QueryResult,
        sql: str,
    ) -> str:
        lines: List[str] = [
            f"QUESTION: {question}",
            "",
            f"SQL EXECUTED: {sql}",
            "",
            f"COLUMNS: {', '.join(result.columns)}",
            f"ROWS RETURNED: {result.row_count}"
            + (" (truncated)" if result.truncated else ""),
            "",
            "RESULTS:",
        ]

        # Include up to 50 rows in the prompt
        for row in result.rows[:50]:
            lines.append(" | ".join(str(v) for v in row))

        if len(result.rows) > 50:
            lines.append(f"... and {len(result.rows) - 50} more rows")

        return "\n".join(lines)

    def _fallback_summary(self, result: QueryResult) -> str:
        if not result.columns:
            return "Query completed with no results."
        header = " | ".join(result.columns)
        sep = " | ".join(["---"] * len(result.columns))
        body = "\n".join(
            " | ".join(str(v) for v in row)
            for row in result.rows[:20]
        )
        return f"Returned {result.row_count} rows:\n\n{header}\n{sep}\n{body}"
