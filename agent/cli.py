from __future__ import annotations

import logging
import os
import sys

from agent.config import get_llm_config
from agent.graph import build_graph
from agent.models import AgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _extract_available_models() -> list[str]:
    """Собирает список доступных моделей из конфига."""

    config = get_llm_config()
    seen: set[str] = set()
    models: list[str] = []

    for node_config in config.nodes.values():
        for model_cfg in node_config.models:
            if model_cfg.model not in seen:
                seen.add(model_cfg.model)
                models.append(model_cfg.model)

    if config.defaults.model not in seen:
        models.append(config.defaults.model)

    return models


def _handle_model_command(raw_query: str) -> tuple[str | None, str]:
    """Обрабатывает команду model=<имя> [запрос]."""

    raw_value = raw_query.split("=", 1)[1].strip()
    parts = raw_value.split(maxsplit=1)

    model_name = parts[0] if parts else ""
    tail_query = parts[1] if len(parts) > 1 else ""

    if model_name:
        os.environ["LLM_PREFERRED_MODEL"] = model_name
        return model_name, tail_query

    os.environ.pop("LLM_PREFERRED_MODEL", None)
    return None, tail_query


def run_cli():
    """CLI для взаимодействия с агентом."""

    app = build_graph()

    print("AI-агент запущен.")
    print("Выход: Ctrl+C / Ctrl+D / пустой ввод.")
    print("Команды:")
    print("  models                — список моделей")
    print("  model=<имя> [запрос]  — выбрать модель")

    while True:
        try:
            query = input("\nВведите запрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nЗавершение работы.")
            break

        if not query:
            print("Пустой ввод. Завершение.")
            break

        if query.lower() == "models":
            for model in _extract_available_models():
                print(f"- {model}")
            continue

        if query.lower().startswith("model="):
            selected_model, tail = _handle_model_command(query)

            if selected_model:
                print(f"Выбрана модель: {selected_model}")
            else:
                print("Сброс prefer-модели")

            if not tail:
                continue

            query = tail

        try:
            state = AgentState(user_query=query)
            result = app.invoke(state)

            if isinstance(result, dict):
                answer = result.get("final_answer") or result.get("intermediate_result")
            else:
                answer = result.final_answer or result.intermediate_result

            print("\nОтвет:\n")
            print(answer or "Ответ не сформирован.")
            print("-" * 50)

        except Exception as exc:
            logger.exception("Ошибка выполнения")
            print(f"\nОшибка: {exc}")