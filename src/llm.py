"""LLM client abstraction supporting Azure OpenAI, OpenAI-compatible endpoints, and Anthropic.

Supports:
- Azure OpenAI (resource.openai.azure.com)
- Local OpenAI-compatible servers (Ollama, LM Studio, vLLM, etc.)
- Anthropic (api.anthropic.com)
"""

from typing import Any, Dict, List, Optional

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore[assignment]

try:
    from openai import AzureOpenAI, OpenAI
    from openai.types.chat import ChatCompletionMessageParam
except ImportError:
    AzureOpenAI = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]
    ChatCompletionMessageParam = Dict[str, Any]  # type: ignore[assignment]

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


PROVIDER_AZURE = "azure"
PROVIDER_LOCAL = "local"
PROVIDER_ANTHROPIC = "anthropic"
ALL_PROVIDERS = [PROVIDER_AZURE, PROVIDER_LOCAL, PROVIDER_ANTHROPIC]


def _format_exception_with_cause(exc: Exception) -> str:
    """Build a concise message including nested cause/context details."""
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


class LLMConfig:
    """Configuration for an LLM connection."""

    def __init__(
        self,
        provider: str = PROVIDER_LOCAL,
        endpoint: str = "",
        model: str = "",
        api_key: str = "",
        api_version: str = "2024-02-01",
        deployment_name: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        verify_ssl: bool = True,
    ) -> None:
        self.provider = provider
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def is_configured(self) -> bool:
        if not self.model:
            return False
        if self.provider == PROVIDER_AZURE:
            return bool(self.endpoint and self.api_key and self.deployment_name)
        if self.provider == PROVIDER_ANTHROPIC:
            return bool(self.endpoint and self.api_key)
        # Local OpenAI-compatible servers don't require an API key.
        return bool(self.endpoint)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key": self.api_key,
            "api_version": self.api_version,
            "deployment_name": self.deployment_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=data.get("provider", PROVIDER_LOCAL),
            endpoint=data.get("endpoint", ""),
            model=data.get("model", ""),
            api_key=data.get("api_key", ""),
            api_version=data.get("api_version", "2024-02-01"),
            deployment_name=data.get("deployment_name", ""),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 16000),
            timeout=data.get("timeout", 120),
            verify_ssl=data.get("verify_ssl", True),
        )


class LLMClient:
    """Client for sending chat completions to various LLM backends."""

    def __init__(self, config: LLMConfig) -> None:
        if not config.is_configured():
            raise ValueError("LLMConfig is not fully configured")
        self.config = config
        self._client: Any = self._create_client()

    def _create_client(self) -> Any:
        http_client = None
        if not self.config.verify_ssl:
            if _httpx is None:
                raise ImportError("httpx package is required to disable SSL verification")
            http_client = _httpx.Client(verify=False)

        if self.config.provider == PROVIDER_AZURE:
            if AzureOpenAI is None:
                raise ImportError("openai package is required for Azure support")
            return AzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                timeout=self.config.timeout,
                max_retries=1,
                http_client=http_client,
            )
        elif self.config.provider == PROVIDER_ANTHROPIC:
            if anthropic is None:
                raise ImportError("anthropic package is required for Anthropic support")
            base_url = self.config.endpoint or "https://api.anthropic.com"
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            return anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=1,
                http_client=http_client,
            )
        else:
            if OpenAI is None:
                raise ImportError("openai package is required")
            base_url = self.config.endpoint or "https://api.openai.com/v1"
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            return OpenAI(
                # Local OpenAI-compatible servers often need no key, but the SDK
                # requires a non-empty value, so fall back to a placeholder.
                api_key=self.config.api_key or "not-needed",
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=1,
                http_client=http_client,
            )

    def _get_model_name(self) -> str:
        if self.config.provider == PROVIDER_AZURE:
            return self.config.deployment_name
        return self.config.model

    def _convert_messages_for_anthropic(
        self,
        messages: List[ChatCompletionMessageParam],
    ) -> tuple[str, List[Dict[str, str]]]:
        """Convert OpenAI-style messages to Anthropic format.

        Returns:
            Tuple of (system prompt, messages list without system role).
        """
        system_prompt = ""
        anthropic_messages: List[Dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle content blocks (list of dicts with 'type' key)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "\n".join(text_parts)

            if not content:
                continue

            if role == "system":
                system_prompt = str(content)
            elif role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": str(content)})
            # Skip 'tool' and other roles Anthropic doesn't support

        return system_prompt, anthropic_messages

    def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return the assistant's text.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            timeout: Override config timeout (seconds).

        Returns:
            The assistant's text response.
        """
        model = self._get_model_name()
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            if self.config.provider == PROVIDER_ANTHROPIC:
                system_prompt, anth_messages = self._convert_messages_for_anthropic(messages)
                if not anth_messages:
                    raise ValueError("No user/assistant messages after conversion")

                response = self._client.messages.create(
                    model=model,
                    messages=anth_messages,
                    system=system_prompt,
                    temperature=temp,
                    max_tokens=tokens,
                )
                content = response.content[0].text  # type: ignore[union-attr]
            else:
                to = timeout if timeout is not None else self.config.timeout
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    timeout=to,
                )
                content = response.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - preserve root cause for UI debugging
            details = _format_exception_with_cause(e)
            raise RuntimeError(f"LLM request failed: {details}") from e

        if not content:
            raise ValueError("LLM returned empty response")
        return content.strip()
