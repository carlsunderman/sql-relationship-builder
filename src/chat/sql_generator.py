"""SQL Generator agent: natural-language question -> T-SQL query."""

import logging
from typing import Any, Dict, List, Optional

from src.chat.context_builder import build_schema_context, parse_sql_generator_response
from src.chat.models import ChatMessage, GeneratedQuery, MessageRole
from src.llm import LLMClient

logger = logging.getLogger(__name__)


class SQLGenerator:
    """Generates SQL Server queries from natural-language questions."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        graph_id: str,
        domain_context: str = "",
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> GeneratedQuery:
        """Generate a SQL query for the given question.

        Args:
            question: The user's natural-language question.
            graph_id: ID of the persisted graph providing schema context.
            domain_context: Optional domain description.
            conversation_history: Optional prior turns for follow-up context.

        Returns:
            GeneratedQuery with the SQL and reasoning.
        """
        ctx = build_schema_context(
            graph_id=graph_id,
            domain_context=domain_context,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": ctx["prompt"]},
        ]

        # Add recent conversation history for follow-up context
        if conversation_history:
            for msg in conversation_history[-6:]:  # last 3 turns (user + assistant pairs)
                messages.append({
                    "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                    "content": msg.content,
                })

        messages.append({"role": "user", "content": question})

        try:
            response = self.llm.chat(messages, temperature=0.0)
        except Exception as e:
            logger.error("SQL generation failed: %s", e)
            return GeneratedQuery(
                sql="",
                reasoning=f"LLM error: {e}",
                target_connection_id="",
            )

        return parse_sql_generator_response(response)
