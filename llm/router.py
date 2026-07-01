import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    provider: str
    cached: bool


class LLMRouter:
    def __init__(self, cache_path: str | None = None, providers: Iterable[str] | None = None) -> None:
        default_cache = Path(__file__).resolve().parents[1] / "demo" / "router_cache.sqlite"
        self.cache_path = Path(cache_path) if cache_path else default_cache
        self.providers = list(providers) if providers else self._provider_list()
        self.timeout_seconds = int(os.getenv("DM_FORGE_TIMEOUT_SECONDS", "30"))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _provider_list(self) -> list[str]:
        raw = os.getenv("DM_FORGE_PROVIDERS", "groq,openrouter,gemini,ollama")
        return [item.strip().lower() for item in raw.split(",") if item.strip()]

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _cache_key(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        sensitive: bool,
    ) -> str:
        body = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "sensitive": sensitive,
        }
        encoded = json.dumps(body, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_cached(self, cache_key: str) -> LLMResponse | None:
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "SELECT response, provider FROM responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return LLMResponse(text=row[0], provider=row[1], cached=True)

    def set_cached(self, cache_key: str, text: str, provider: str) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO responses (cache_key, response, provider)
                VALUES (?, ?, ?)
                """,
                (cache_key, text, provider),
            )
            conn.commit()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 300,
        sensitive: bool = False,
    ) -> LLMResponse:
        cache_key = self._cache_key(prompt, system_prompt, temperature, max_tokens, sensitive)
        cached = self.get_cached(cache_key)
        if cached:
            return cached

        provider_order = self._ordered_providers(sensitive=sensitive)
        errors: list[str] = []
        for provider in provider_order:
            try:
                text = self._call_provider(
                    provider=provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.set_cached(cache_key, text, provider)
                return LLMResponse(text=text, provider=provider, cached=False)
            except Exception as exc:
                errors.append(f"{provider}: {exc}")

        if os.getenv("DM_FORGE_ALLOW_OFFLINE_FALLBACK", "0") == "1":
            fallback = self._offline_response(prompt)
            self.set_cached(cache_key, fallback, "offline")
            return LLMResponse(text=fallback, provider="offline", cached=False)

        raise RuntimeError("All providers failed: " + " | ".join(errors))

    def _ordered_providers(self, sensitive: bool) -> list[str]:
        if sensitive and "ollama" in self.providers:
            remaining = [p for p in self.providers if p != "ollama"]
            return ["ollama", *remaining]
        return self.providers

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if provider == "groq":
            return self._call_openai_compatible(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY", ""),
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openrouter":
            return self._call_openai_compatible(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "gemini":
            return self._call_gemini(
                api_key=os.getenv("GEMINI_API_KEY", ""),
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "ollama":
            return self._call_ollama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise ValueError(f"Unknown provider: {provider}")

    def _call_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not api_key:
            raise ValueError("Missing API key")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout_seconds)
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("Empty response")
        return text.strip()

    def _call_gemini(
        self,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not api_key:
            raise ValueError("Missing API key")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        text = f"{system_prompt}\n\n{prompt}".strip()
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        response = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        first = candidates[0] if candidates else {}
        content = first.get("content") or {}
        parts = content.get("parts") or []
        first_part = parts[0] if parts else {}
        output = first_part.get("text")
        if not output:
            raise ValueError("Empty response")
        return output.strip()

    def _call_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        output = data.get("response")
        if not output:
            raise ValueError("Empty response")
        return output.strip()

    def _offline_response(self, prompt: str) -> str:
        compact = " ".join(prompt.split())
        if len(compact) <= 300:
            return compact
        return compact[:297] + "..."
