from __future__ import annotations

import os
import time
from typing import Any, Final

from dotenv import load_dotenv
from openai import OpenAI

from src.common.config import ROOT

PROVIDERS: Final[dict[str, dict[str, str]]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-3.5-turbo",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-3.5-turbo",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen2.5-coder-32b-instruct",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url_env": "GROQ_BASE_URL",
        "default_base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.5-flash",
    },
}


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=True)


def provider_config(provider: str) -> dict[str, str]:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}") from exc


def provider_keys() -> list[str]:
    return sorted(PROVIDERS)


def default_model(provider: str) -> str:
    return provider_config(provider)["default_model"]


def default_headers(provider: str) -> dict[str, str] | None:
    if provider != "openrouter":
        return None

    headers = {
        key: value
        for key, value in {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL"),
            "X-OpenRouter-Title": os.getenv("OPENROUTER_SITE_NAME"),
        }.items()
        if value
    }

    return headers or None


def get_client(provider: str) -> OpenAI:
    load_environment()

    config = provider_config(provider)
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise RuntimeError(f"{config['api_key_env']} not found in .env")

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": os.getenv(config["base_url_env"], config["default_base_url"]),
    }

    headers = default_headers(provider)
    if headers:
        kwargs["default_headers"] = headers

    return OpenAI(**kwargs)


def usage_stats(response: Any, latency: float) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "latency_seconds": round(latency, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_second": (
            round(completion_tokens / latency, 4)
            if completion_tokens and latency > 0
            else None
        ),
    }


def make_request(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        request["response_format"] = {"type": "json_object"}

    return request


def call_chat_model(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    request = make_request(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )

    started = time.perf_counter()

    try:
        response = client.chat.completions.create(**request)
    except Exception:
        if not json_mode:
            raise
        request.pop("response_format", None)
        response = client.chat.completions.create(**request)

    latency = time.perf_counter() - started
    content = response.choices[0].message.content or ""

    return content, usage_stats(response, latency)



