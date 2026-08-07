"""
llm_client.py
=============
Thin wrapper around an OpenAI-compatible Chat Completions endpoint.

Works with:
  - OpenAI API            (https://api.openai.com/v1)
  - Azure OpenAI          (set base_url to your resource endpoint)
  - Local Ollama          (base_url="http://localhost:11434/v1", api_key="ollama")
  - LM Studio             (base_url="http://localhost:1234/v1")

All configuration is read from environment variables / passed explicitly so the
Streamlit sidebar can override them at runtime without editing code.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass


@dataclass
class LLMConfig:
    base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key: str = os.environ.get("LLM_API_KEY", "ollama")
    model: str = os.environ.get("LLM_MODEL", "llama3.1:latest")
    temperature: float = 0.3


def resolve_config_from_secrets(default: "LLMConfig") -> "LLMConfig":
    """On Streamlit Community Cloud, prefer st.secrets (set in the app's
    'Secrets' panel) over the local Ollama defaults, since a hosted app has
    no access to localhost:11434 on the developer's machine.

    Falls back to `default` untouched when secrets aren't configured (e.g.
    running locally with no .streamlit/secrets.toml).

    Accepts a few common key-name variants so a small naming mismatch in the
    Secrets panel doesn't silently fall back to the (unreachable, on a
    hosted server) local Ollama default.
    """
    try:
        import streamlit as st
        base_url = (st.secrets.get("LLM_BASE_URL") or st.secrets.get("GROQ_BASE_URL")
                    or st.secrets.get("OPENAI_BASE_URL"))
        api_key = (st.secrets.get("LLM_API_KEY") or st.secrets.get("GROQ_API_KEY")
                   or st.secrets.get("OPENAI_API_KEY") or st.secrets.get("API_KEY"))
        model = st.secrets.get("LLM_MODEL") or st.secrets.get("GROQ_MODEL")
        if api_key:
            return LLMConfig(
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key,
                model=model or "gpt-4o-mini",
                temperature=default.temperature,
            )
    except Exception:
        pass
    return default


def is_unreachable_local_config(config: "LLMConfig") -> bool:
    """True if this config points at a localhost/loopback address -- which
    can never be reached from a hosted server (Streamlit Community Cloud,
    etc). Used to surface a clear error instead of a raw APIConnectionError.
    """
    host = config.base_url.lower()
    return ("localhost" in host) or ("127.0.0.1" in host) or ("0.0.0.0" in host)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        # Imported lazily so the app can still boot without the package
        # installed until the user actually runs an LLM call.
        from openai import OpenAI
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key or "not-needed")

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> dict:
        """Call the chat endpoint and parse the response as JSON.

        Falls back to extracting the first {...} block if the model wraps
        the JSON in prose or markdown fences (common with local models that
        don't support strict JSON mode).
        """
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception:
            # Some backends (older Ollama/LM Studio builds) reject response_format.
            resp = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        content = resp.choices[0].message.content or ""
        return _coerce_json(content)


def _coerce_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: grab the largest {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")
