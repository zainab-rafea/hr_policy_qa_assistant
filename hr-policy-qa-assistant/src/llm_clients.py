"""Thin free LLM clients (HTTP) — no paid Anthropic key required."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _messages_to_text(messages: list[BaseMessage]) -> tuple[str | None, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, SystemMessage):
            system_parts.append(content)
        elif isinstance(msg, HumanMessage):
            user_parts.append(content)
        else:
            user_parts.append(content)
    system = "\n\n".join(system_parts) if system_parts else None
    user = "\n\n".join(user_parts)
    return system, user


class ChatGeminiHTTP(BaseChatModel):
    """Google Gemini via the free Generative Language API."""

    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    api_key: str | None = None
    timeout: float = 120.0

    @property
    def _llm_type(self) -> str:
        return "gemini-http"

    def _resolve_key(self) -> str:
        key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Missing GOOGLE_API_KEY. Get a free key at https://aistudio.google.com/apikey"
            )
        return key

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, user = _messages_to_text(messages)
        key = self._resolve_key()
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        contents = [{"role": "user", "parts": [{"text": user}]}]
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, params={"key": key}, json=body)
            resp.raise_for_status()
            data = resp.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from exc

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class ChatOllamaHTTP(BaseChatModel):
    """Local Ollama chat API (fully free / offline after model pull)."""

    model: str = "llama3.2:3b"
    temperature: float = 0.0
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 300.0
    format: str | None = "json"

    @property
    def _llm_type(self) -> str:
        return "ollama-http"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload_messages = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            payload_messages.append({"role": role, "content": content})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.format:
            body["format"] = self.format

        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(f"{self.base_url}/api/chat", json=body)
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    "Cannot reach Ollama at http://127.0.0.1:11434. "
                    "Start the Ollama app, then run: ollama pull llama3.2:3b"
                ) from exc
            data = resp.json()

        text = data.get("message", {}).get("content", "")
        if isinstance(text, dict):
            text = json.dumps(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
