from __future__ import annotations

import logging
from typing import Any, Callable

from agent.chains import (
    build_fallback_chain,
    build_intent_chain,
    build_search_fallback_chain,
    build_sentiment_chain,
    build_summary_chain,
    fetch_plain_text,
    run_tavily_search,
)
from agent.config import get_tavily_api_key, iter_llms_for_node
from agent.models import AgentState, IntentRecognitionOutput, IntentType

logger = logging.getLogger(__name__)


def _run_node_with_fallback(
    node_name: str,
    chain_builder: Callable[[Any], Any],
    payload: dict[str, Any],
):
    """Выполняет шаг обработки с поочерёдным переключением между моделями."""

    last_error: Exception | None = None

    for model_name, llm in iter_llms_for_node(node_name):
        try:
            chain = chain_builder(llm)
            result = chain.invoke(payload)
            logger.info("Шаг '%s' выполнен с использованием модели '%s'", node_name, model_name)
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Сбой при выполнении шага '%s' на модели '%s': %s",
                node_name,
                model_name,
                exc,
            )

    raise RuntimeError(
        f"Не удалось завершить шаг '{node_name}'. Последняя ошибка: {last_error}"
    )


def normalize_intents(raw: list[str]) -> list[IntentType]:
    """Приводит список интентов к допустимому и предсказуемому виду."""

    seen: set[str] = set()
    intents: list[IntentType] = []

    for intent in raw:
        if intent not in ("search", "summarize", "sentiment", "none"):
            continue
        if intent in seen:
            continue
        seen.add(intent)
        intents.append(intent)

    if len(intents) > 1 and "none" in intents:
        intents.remove("none")

    try:
        search_idx = intents.index("search")
        summarize_idx = intents.index("summarize")
        if search_idx > summarize_idx:
            intents.pop(search_idx)
            intents.insert(summarize_idx, "search")
    except ValueError:
        pass

    if "search" in intents and "sentiment" in intents and "summarize" not in intents:
        sentiment_idx = intents.index("sentiment")
        intents.insert(sentiment_idx, "summarize")

    return intents or ["none"]


def merge_step_output(state: AgentState, title: str, content: str) -> str:
    """Добавляет результат текущего шага к уже накопленному ответу."""

    clean_content = content.replace("**", "").strip()
    parts: list[str] = []

    if state.final_answer:
        parts.append(state.final_answer.strip())

    parts.append(f"{title}:\n{clean_content}")
    return "\n\n".join(parts)


def intent_recognition_node(state: AgentState) -> AgentState:
    try:
        prediction = _run_node_with_fallback(
            node_name="intent",
            chain_builder=build_intent_chain,
            payload={"user_query": state.user_query},
        )

        if isinstance(prediction, IntentRecognitionOutput):
            intents = normalize_intents(prediction.intents)
        else:
            intents = ["none"]

        logger.info("Распознанные интенты: %s", intents)

        state.detected_intents = intents
        state.execution_chain = list(intents)
        state.intermediate_result = state.user_query
        state.final_answer = None
        state.next_intent_index = 0

    except Exception as exc:
        logger.error("Не удалось распознать интенты: %s", exc)
        error_msg = f"Не удалось распознать интенты. Детали: {exc}"

        state.detected_intents = ["none"]
        state.execution_chain = ["none"]
        state.intermediate_result = error_msg
        state.final_answer = error_msg
        state.next_intent_index = 0

    return state


def search_tool_handler(state: AgentState) -> AgentState:
    query = state.intermediate_result or state.user_query

    try:
        _ = get_tavily_api_key()
        payload, urls = run_tavily_search(query)

        if not payload or payload.strip() == "Поиск не вернул структурированных результатов.":
            raise RuntimeError("Tavily не вернул содержательных результатов.")

        page_blocks: list[str] = []

        for url in urls[:3]:
            page_text = fetch_plain_text(url)
            if page_text:
                page_blocks.append(f"Источник: {url}\n{page_text}")

        if page_blocks:
            payload = payload + "\n\n" + "\n\n".join(page_blocks)

    except Exception as exc:
        logger.warning(
            "Tavily-поиск недоступен или не дал результатов, пробуем резервный сценарий: %s",
            exc,
        )
        try:
            response = _run_node_with_fallback(
                node_name="search",
                chain_builder=build_search_fallback_chain,
                payload={"query": query},
            )
            payload = getattr(response, "content", str(response))
        except Exception as fallback_exc:
            logger.error("Резервный сценарий поиска тоже завершился ошибкой: %s", fallback_exc)
            payload = f"Ошибка при выполнении поиска: {fallback_exc}"

    state.intermediate_result = payload
    state.final_answer = state.final_answer or ""
    state.next_intent_index += 1
    return state


def summarize_handler(state: AgentState) -> AgentState:
    content = state.intermediate_result or state.user_query

    try:
        response = _run_node_with_fallback(
            node_name="summarize",
            chain_builder=build_summary_chain,
            payload={"content": content},
        )
        payload = getattr(response, "content", str(response)).strip()

        if payload.lower().startswith("краткое резюме"):
            payload = payload.split(":", 1)[-1].strip()

    except Exception as exc:
        logger.error("Ошибка суммаризации: %s", exc)
        payload = "Не удалось выполнить суммаризацию."

    state.intermediate_result = payload

    if "sentiment" in state.detected_intents:
        state.final_answer = state.final_answer or ""
    else:
        state.final_answer = merge_step_output(state, "Краткое резюме", payload)

    state.next_intent_index += 1
    return state


def sentiment_handler(state: AgentState) -> AgentState:
    content = state.intermediate_result or state.user_query

    try:
        sentiment = _run_node_with_fallback(
            node_name="sentiment",
            chain_builder=build_sentiment_chain,
            payload={"content": content},
        )
        payload = getattr(sentiment, "content", str(sentiment))
    except Exception as exc:
        logger.error("Не удалось выполнить оценку тональности: %s", exc)
        payload = "Не удалось выполнить оценку тональности."

    state.intermediate_result = payload
    state.final_answer = merge_step_output(state, "Оценка тональности", payload)
    state.next_intent_index += 1
    return state


def fallback_handler(state: AgentState) -> AgentState:
    try:
        reply = _run_node_with_fallback(
            node_name="fallback",
            chain_builder=build_fallback_chain,
            payload={"query": state.user_query},
        )
        payload = getattr(reply, "content", str(reply))
    except Exception as exc:
        logger.error("Не удалось сформировать прямой ответ: %s", exc)
        payload = "Не удалось сформировать прямой ответ."

    state.intermediate_result = payload
    state.detected_intents = state.detected_intents or ["none"]
    state.final_answer = payload
    state.next_intent_index += 1
    return state


def finalize_node(state: AgentState) -> AgentState:
    if not state.final_answer:
        state.final_answer = state.intermediate_result or "Итоговый ответ не был сформирован."
    return state