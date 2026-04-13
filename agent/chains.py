from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.runnables import RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults

from agent.models import IntentRecognitionOutput, SearchResult, SearchResults
from agent.prompts import (
    FALLBACK_PROMPT,
    INTENT_PROMPT,
    SEARCH_FALLBACK_PROMPT,
    SENTIMENT_PROMPT,
    SUMMARIZE_PROMPT,
)


def _parse_intents(output: Any) -> IntentRecognitionOutput:
    """Извлекает и нормализует список интентов из ответа LLM."""

    text = getattr(output, "content", None) or str(output)

    intents: list[str] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            intents = [str(item) for item in data.get("intents", [])]
    except Exception:
        pass

    allowed = {"search", "summarize", "sentiment", "none"}
    filtered: list[str] = []
    seen = set()

    for intent in intents:
        if intent in allowed and intent not in seen:
            filtered.append(intent)
            seen.add(intent)

    if not filtered:
        filtered = ["none"]

    priority = {"search": 0, "summarize": 1, "sentiment": 2, "none": 99}
    if len(filtered) > 1 and "none" in filtered:
        filtered.remove("none")

    filtered = sorted(filtered, key=lambda item: priority[item])

    return IntentRecognitionOutput(intents=filtered)


def _normalize_search_results(raw_result: Any) -> SearchResults:
    items: list[SearchResult] = []

    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, dict):
                items.append(
                    SearchResult(
                        title=str(item.get("title", "")),
                        url=str(item.get("url", "")),
                        snippet=str(item.get("content") or item.get("snippet") or ""),
                    )
                )

    elif isinstance(raw_result, dict):
        raw_items = raw_result.get("results", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    items.append(
                        SearchResult(
                            title=str(item.get("title", "")),
                            url=str(item.get("url", "")),
                            snippet=str(item.get("content") or item.get("snippet") or ""),
                        )
                    )

    elif hasattr(raw_result, "results"):
        raw_items = getattr(raw_result, "results", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    items.append(
                        SearchResult(
                            title=str(item.get("title", "")),
                            url=str(item.get("url", "")),
                            snippet=str(item.get("content") or item.get("snippet") or ""),
                        )
                    )

    return SearchResults(results=items)


def _extract_top_urls(raw_result: Any) -> list[str]:
    """Извлекает URL из search-результатов."""

    urls: list[str] = []

    if isinstance(raw_result, list):
        for item in raw_result:
            if isinstance(item, dict):
                url = str(item.get("url", "")).strip()
                if url and url not in urls:
                    urls.append(url)

    elif isinstance(raw_result, dict):
        raw_items = raw_result.get("results", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    url = str(item.get("url", "")).strip()
                    if url and url not in urls:
                        urls.append(url)

    elif hasattr(raw_result, "results"):
        raw_items = getattr(raw_result, "results", [])
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    url = str(item.get("url", "")).strip()
                    if url and url not in urls:
                        urls.append(url)

    return urls


def _shorten_snippet(text: str, max_length: int = 500) -> str:
    """Ограничивает длину сниппета и слегка нормализует пробелы."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized

    return normalized[: max_length - 3].rstrip() + "..."


def _render_search_results(results: SearchResults) -> str:
    if not results.results:
        return "Поиск не вернул структурированных результатов."

    return "\n\n".join(
        [
            f"{idx}. {item.title}\n"
            f"URL: {item.url}\n"
            f"Snippet: {_shorten_snippet(item.snippet)}"
            for idx, item in enumerate(results.results, start=1)
        ]
    )


def fetch_plain_text(url: str, max_bytes: int = 200_000, timeout: int = 10) -> str:
    """Загружает HTML-страницу и извлекает из неё текст."""

    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=timeout) as response:
            content_type = response.info().get_content_type()
            if content_type != "text/html":
                return ""

            raw = response.read(max_bytes)

        decoded = raw.decode("utf-8", errors="ignore")
        decoded = re.sub(r"(?is)<script.*?>.*?</script>", " ", decoded)
        decoded = re.sub(r"(?is)<style.*?>.*?</style>", " ", decoded)
        decoded = re.sub(r"<[^>]+>", " ", decoded)
        decoded = re.sub(r"\s+", " ", decoded).strip()

        return decoded[:8000]
    except (HTTPError, URLError, TimeoutError, ValueError):
        return ""


def build_intent_chain(llm):
    return INTENT_PROMPT | llm | RunnableLambda(_parse_intents)


def build_summary_chain(llm):
    return SUMMARIZE_PROMPT | llm


def build_sentiment_chain(llm):
    return SENTIMENT_PROMPT | llm


def build_fallback_chain(llm):
    return FALLBACK_PROMPT | llm


def build_search_fallback_chain(llm):
    return SEARCH_FALLBACK_PROMPT | llm


def run_tavily_search(query: str) -> tuple[str, list[str]]:
    search_tool = TavilySearchResults(max_results=5)
    raw_result = search_tool.invoke(query)

    normalized = _normalize_search_results(raw_result)
    rendered = _render_search_results(normalized)
    urls = _extract_top_urls(raw_result)

    return rendered, urls