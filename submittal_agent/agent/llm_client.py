"""Unified LLM client with automatic fallback between Claude and GPT-4."""

import json
import logging
from typing import Optional, Any

from anthropic import Anthropic, APIError, RateLimitError
from openai import OpenAI

from submittal_agent.config import (
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    LLMConfig,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client with automatic fallback.

    Primary: Claude Opus 4.5 (claude-opus-4-5-20251101)
    Fallback: GPT-5.2

    Supports both text and vision (multimodal) requests.
    """

    def __init__(self):
        self.claude = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self._last_provider: Optional[str] = None

    @property
    def last_provider(self) -> str:
        """Get the provider used for the last request."""
        return self._last_provider or "none"

    def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = LLMConfig.PRIMARY_MAX_TOKENS,
        temperature: float = LLMConfig.TEMPERATURE,
        json_mode: bool = False,
        model: str = "auto",
    ) -> tuple[str, str]:
        """
        Complete a text request with automatic fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_mode: If True, request JSON output
            model: Model selection - "auto", "claude", or "gpt-4"

        Returns:
            Tuple of (response_text, provider_used)
        """
        # Handle explicit model selection
        if model == "gpt-5":
            if self.openai:
                try:
                    response = self._call_openai(
                        messages=messages,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        json_mode=json_mode,
                    )
                    self._last_provider = "gpt-5.2"
                    return response, "gpt-5.2"
                except Exception as e:
                    logger.error(f"GPT-5.2 failed: {e}")
                    raise
            raise ValueError("GPT-5.2 selected but OPENAI_API_KEY not set.")

        if model == "claude":
            if self.claude:
                try:
                    response = self._call_claude(
                        messages=messages,
                        system=system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    self._last_provider = "claude"
                    return response, "claude"
                except Exception as e:
                    logger.error(f"Claude failed: {e}")
                    raise
            raise ValueError("Claude selected but ANTHROPIC_API_KEY not set.")

        # Auto mode: Try Claude first with GPT-4 fallback
        if self.claude:
            try:
                response = self._call_claude(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._last_provider = "claude"
                return response, "claude"
            except (RateLimitError, APIError) as e:
                logger.warning(f"Claude failed: {e}, falling back to GPT-4")

        # Fallback to GPT-5.2
        if self.openai:
            try:
                response = self._call_openai(
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                )
                self._last_provider = "gpt-5.2"
                return response, "gpt-5.2"
            except Exception as e:
                logger.error(f"GPT-5.2 also failed: {e}")
                raise

        raise ValueError("No LLM provider available. Check API keys.")

    def complete_with_vision(
        self,
        text_prompt: str,
        images_base64: list[dict],
        system: Optional[str] = None,
        max_tokens: int = LLMConfig.VISION_MAX_TOKENS,
    ) -> tuple[str, str]:
        """
        Complete a vision request (multimodal).

        Args:
            text_prompt: Text prompt to accompany images
            images_base64: List of dicts with 'type', 'source' (base64 image data)
            system: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, provider_used)
        """
        # Try Claude Vision first
        if self.claude:
            try:
                response = self._call_claude_vision(
                    text_prompt=text_prompt,
                    images=images_base64,
                    system=system,
                    max_tokens=max_tokens,
                )
                self._last_provider = "claude-vision"
                return response, "claude-vision"
            except (RateLimitError, APIError) as e:
                logger.warning(f"Claude Vision failed: {e}, falling back to GPT-5.2 Vision")

        # Fallback to GPT-5.2 Vision
        if self.openai:
            try:
                response = self._call_openai_vision(
                    text_prompt=text_prompt,
                    images=images_base64,
                    system=system,
                    max_tokens=max_tokens,
                )
                self._last_provider = "gpt-5.2-vision"
                return response, "gpt-5.2-vision"
            except Exception as e:
                logger.error(f"GPT-5.2 Vision also failed: {e}")
                raise

        raise ValueError("No vision-capable LLM available. Check API keys.")

    def _call_claude(
        self,
        messages: list[dict],
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call Claude API."""
        response = self.claude.messages.create(
            model=LLMConfig.PRIMARY_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=messages,
        )
        return response.content[0].text

    def _call_claude_vision(
        self,
        text_prompt: str,
        images: list[dict],
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        """Call Claude Vision API."""
        # Build content with images and text
        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": img["data"],
                }
            })
        content.append({"type": "text", "text": text_prompt})

        response = self.claude.messages.create(
            model=LLMConfig.VISION_MODEL,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant analyzing images.",
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def _call_openai(
        self,
        messages: list[dict],
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """Call OpenAI API."""
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)

        kwargs = {
            "model": LLMConfig.FALLBACK_MODEL,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _call_openai_vision(
        self,
        text_prompt: str,
        images: list[dict],
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        """Call OpenAI Vision API."""
        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.get('media_type', 'image/png')};base64,{img['data']}"
                }
            })
        content.append({"type": "text", "text": text_prompt})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        response = self.openai.chat.completions.create(
            model=LLMConfig.FALLBACK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
