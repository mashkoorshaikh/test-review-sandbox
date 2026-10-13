"""
llm_config.py - LLM Provider Configuration and Client Factory.

Loads API credentials from environment variables using python-dotenv.
Supports standard OpenAI models as well as any OpenAI-compatible provider
(Azure, Ollama, OpenRouter, vLLM, etc.).

Safety & Graceful Degradation:
  - Validates API key presence before attempting to build LLM clients.
  - Never logs or exposes credentials.
  - Raises MissingApiKeyError with clear actionable guidance when credentials
    are absent, allowing the pipeline to degrade gracefully to deterministic scanning.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Load environment variables from .env if present
load_dotenv()

PLACEHOLDER_KEYS = {
    "your_openai_api_key_here",
    "placeholder",
    "none",
    "sk-placeholder",
}


class MissingApiKeyError(RuntimeError):
    """Raised when required LLM credentials are not configured."""
    pass


def is_api_key_configured() -> bool:
    """
    Check whether a non-empty, non-placeholder API key is set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return False
    if api_key.lower() in PLACEHOLDER_KEYS:
        return False
    return True


def get_configured_model_name(override_model: Optional[str] = None) -> str:
    """Return the configured model name or default to 'gpt-4o-mini'."""
    if override_model:
        return override_model
    return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"


def get_configured_timeout() -> int:
    """Return timeout in seconds for LLM calls (default: 30)."""
    raw_val = os.getenv("LLM_TIMEOUT", "30").strip()
    try:
        return max(5, int(raw_val))
    except ValueError:
        return 30


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[int] = None,
) -> BaseChatModel:
    """
    Factory to construct an authenticated ChatOpenAI instance.

    Raises:
        MissingApiKeyError: If OPENAI_API_KEY is missing or contains a placeholder.
    """
    if not is_api_key_configured():
        raise MissingApiKeyError(
            "LLM API credentials are missing or unconfigured.\n"
            "To enable AI-assisted review, create a '.env' file based on '.env.example' "
            "and set OPENAI_API_KEY to your API key, or export OPENAI_API_KEY in your environment."
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = get_configured_model_name(model)
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    request_timeout = timeout or get_configured_timeout()

    kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
        "timeout": request_timeout,
        "max_retries": 2,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)
