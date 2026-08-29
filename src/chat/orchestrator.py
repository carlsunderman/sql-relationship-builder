"""Chat orchestrator — wires the three agents together."""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.chat.answer_synthesizer import AnswerSynthesizer
from src.chat.models import (
    ChatMessage,
    ChatTurn,
    GeneratedQuery,
    MessageRole,
    QueryResult,
)
from src.chat.sql_executor import SQLExecutor
from src.chat.sql_generator import SQLGenerator
from src.llm import LLMClient

logger = logging.getLogger(__name__)

# Maximum number of prior turns to retain in conversation memory
MAX_HISTORY_TURNS = 10


class ChatOrchestrator:
    """Coordinates the SQL Generator, Executor, and Answer Synthesizer agents."""

    def __init__(
        self,
        llm: LLMClient,
        row_limit: int = 1000,
        query_timeout: int = 30,
    ) -> None:
        self.llm = llm
        self.generator = SQLGenerator(llm)
        self.executor = SQLExecutor(row_limit=row_limit, timeout=query_timeout)
        self.synthesizer = AnswerSynthesizer(llm)

    def ask(
        self,
        question: str,
        graph_id: str,
        connection: Any,
        domain_context: str = "",
        history: Optional[List[ChatMessage]] = None,
    ) -> ChatTurn:
        """Run the full pipeline for one user question.

        Args:
            question: The user's natural-language question.
            graph_id: ID of the persisted graph providing schema context.
            connection: Live pyodbc connection to execute against.
            domain_context: Optional domain description.
            history: Prior ChatMessages for multi-turn context.

        Returns:
            ChatTurn with generated SQL, query result, and answer.
        """
        turn = ChatTurn(
            user_question=question,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1: Generate SQL
        try:
            generated: GeneratedQuery = self.generator.generate(
                question=question,
                graph_id=graph_id,
                domain_context=domain_context,
                conversation_history=history or [],
            )
            turn.generated_sql = generated.sql
            turn.sql_reasoning = generated.reasoning
        except Exception as e:
            logger.error("SQL generation step failed: %s", e)
            turn.error = f"SQL generation failed: {e}"
            turn.answer = (
                "I couldn't generate a SQL query for that question. "
                "Please try rephrasing it."
            )
            return turn

        if not generated.sql:
            turn.error = "No SQL generated"
            turn.answer = generated.reasoning or "I couldn't generate a SQL query."
            return turn

        # Step 2: Execute SQL
        result: QueryResult = self.executor.execute(generated.sql, connection)
        turn.query_result = result

        if result.status.value == "blocked":
            turn.error = result.error_message
            turn.answer = (
                f"I can't run that query — {result.error_message}. "
                "Please ask a read-only question about the data."
            )
            return turn

        # Step 3: Synthesize answer
        try:
            answer = self.synthesizer.synthesize(
                question=question,
                query_result=result,
                generated_sql=generated.sql,
            )
            turn.answer = answer
        except Exception as e:
            logger.error("Answer synthesis failed: %s", e)
            turn.answer = (
                f"Query succeeded but I couldn't summarize the results: {e}"
            )

        return turn


def build_chat_messages(turns: List[ChatTurn]) -> List[ChatMessage]:
    """Convert a list of ChatTurns into ChatMessages for conversation history."""
    messages: List[ChatMessage] = []
    for turn in turns:
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=turn.user_question,
            timestamp=turn.timestamp,
        ))
        if turn.answer:
            messages.append(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=turn.answer,
                timestamp=turn.timestamp,
            ))
    return messages


def prune_history(
    messages: List[ChatMessage],
    max_turns: int = MAX_HISTORY_TURNS,
) -> List[ChatMessage]:
    """Keep only the last N turns (each turn = user + assistant pair)."""
    max_messages = max_turns * 2
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]
