from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent.models import LLMConfig

load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "defaults": {
        "model": "meta-llama/llama-3-70b-instruct",
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 1024,
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "nodes": {
        "intent": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 256,
                    "base_url": os.getenv(
                        "CLOUDRU_BASE_URL",
                        "https://foundation-models.api.cloud.ru/v1",
                    ),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {
                    "model": "meta-llama/llama-3-70b-instruct",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 256,
                },
            ]
        },
        "search": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "max_tokens": 512,
                    "base_url": os.getenv(
                        "CLOUDRU_BASE_URL",
                        "https://foundation-models.api.cloud.ru/v1",
                    ),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {
                    "model": "meta-llama/llama-3-70b-instruct",
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "max_tokens": 512,
                },
            ]
        },
        "summarize": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "base_url": os.getenv(
                        "CLOUDRU_BASE_URL",
                        "https://foundation-models.api.cloud.ru/v1",
                    ),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {
                    "model": "meta-llama/llama-3-70b-instruct",
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                },
            ]
        },
        "sentiment": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "max_tokens": 256,
                    "base_url": os.getenv(
                        "CLOUDRU_BASE_URL",
                        "https://foundation-models.api.cloud.ru/v1",
                    ),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {
                    "model": "meta-llama/llama-3-70b-instruct",
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "max_tokens": 256,
                },
            ]
        },
        "fallback": {
            "models": [
                {
                    "model": "ai-sage/GigaChat3-10B-A1.8B",
                    "temperature": 0.5,
                    "top_p": 1.0,
                    "max_tokens": 1200,
                    "base_url": os.getenv(
                        "CLOUDRU_BASE_URL",
                        "https://foundation-models.api.cloud.ru/v1",
                    ),
                    "api_key_env": "CLOUDRU_API_KEY",
                },
                {
                    "model": "meta-llama/llama-3-70b-instruct",
                    "temperature": 0.6,
                    "top_p": 1.0,
                    "max_tokens": 1200,
                },
            ]
        },
    },
}


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Файл LLM-конфигурации %s не найден, используются значения по умолчанию.", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Корень LLM-конфига должен быть словарём.")

        return data
    except Exception as exc:
        logger.warning(
            "Не удалось загрузить LLM-конфиг из %s: %s. Используются значения по умолчанию.",
            path,
            exc,
        )
        return {}


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)

    override_defaults = override.get("defaults")
    if isinstance(override_defaults, dict):
        config.setdefault("defaults", {}).update(override_defaults)

    override_nodes = override.get("nodes")
    if isinstance(override_nodes, dict):
        for node_name, node_cfg in override_nodes.items():
            if not isinstance(node_cfg, dict):
                continue

            models = node_cfg.get("models")
            if isinstance(models, list) and models:
                config.setdefault("nodes", {})[node_name] = {"models": models}

    return config


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    config_path = Path(os.getenv("LLM_CONFIG_PATH", "llm_config.yaml"))
    merged = _merge_config(DEFAULT_LLM_CONFIG, _load_yaml_config(config_path))
    return LLMConfig.model_validate(merged)


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _build_chat_llm(
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    base_url: str | None,
    api_key: str,
) -> ChatOpenAI:
    if not api_key:
        raise RuntimeError("API-ключ не найден.")

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def iter_llms_for_node(node_name: str) -> Iterator[tuple[str, ChatOpenAI]]:
    config = get_llm_config()

    defaults = config.defaults.model_dump()
    node_cfg = config.nodes.get(node_name)
    models = [m.model_dump() for m in node_cfg.models] if node_cfg and node_cfg.models else []

    if not models:
        fallback_model = defaults.get("model")
        logger.warning(
            "Для шага '%s' не заданы модели, используется модель по умолчанию '%s'.",
            node_name,
            fallback_model,
        )
        models = [{"model": fallback_model}]

    preferred_model = os.getenv("LLM_PREFERRED_MODEL")
    if preferred_model:
        prioritized: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []

        for entry in models:
            if entry.get("model") == preferred_model:
                prioritized.append(entry)
            else:
                rest.append(entry)

        models = prioritized + rest

    for model_entry in models:
        merged = {**defaults, **(model_entry or {})}

        model_name = merged.get("model")
        if not model_name:
            continue

        temperature = _as_float(
            merged.get("temperature"),
            _as_float(defaults.get("temperature"), 0.3),
        )
        top_p = _as_float(
            merged.get("top_p"),
            _as_float(defaults.get("top_p"), 1.0),
        )
        max_tokens = _as_int(
            merged.get("max_tokens"),
            _as_int(defaults.get("max_tokens"), 1024),
        )
        base_url = merged.get("base_url") or defaults.get("base_url")
        api_key_env = merged.get("api_key_env") or defaults.get("api_key_env")

        if not api_key_env:
            logger.warning(
                "Для модели '%s' в шаге '%s' не указан api_key_env. Модель пропущена.",
                model_name,
                node_name,
            )
            continue

        api_key = os.getenv(api_key_env)

        if not api_key:
            logger.info(
                "Модель '%s' для шага '%s' пропущена: не задана переменная окружения %s.",
                model_name,
                node_name,
                api_key_env,
            )
            continue

        try:
            llm = _build_chat_llm(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
            )
            yield model_name, llm
        except Exception as exc:
            logger.warning(
                "Не удалось инициализировать модель '%s' для шага '%s': %s",
                model_name,
                node_name,
                exc,
            )
            continue


def get_tavily_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Переменная окружения TAVILY_API_KEY не задана.")
    return api_key