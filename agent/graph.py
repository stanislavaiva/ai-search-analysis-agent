from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agent.handlers import (
    fallback_handler,
    finalize_node,
    intent_recognition_node,
    search_tool_handler,
    sentiment_handler,
    summarize_handler,
)
from agent.models import AgentState

logger = logging.getLogger(__name__)


def router_node(state: AgentState) -> AgentState:
    """Техническая нода-маршрутизатор. Само состояние не изменяет."""
    return state


def route_next(state: AgentState) -> str:
    """Определяет следующий шаг на основе текущего индекса и execution chain."""

    execution_chain = state.execution_chain or state.detected_intents

    if state.next_intent_index >= len(execution_chain):
        return "finalize"

    current_intent = execution_chain[state.next_intent_index]

    route_map = {
        "search": "search",
        "summarize": "summarize",
        "sentiment": "sentiment",
        "none": "fallback",
    }

    return route_map.get(current_intent, "fallback")


def build_graph():
    """Собирает граф агента с маршрутизацией по интентам через состояние."""

    graph = StateGraph(AgentState)

    graph.add_node("intent_recognition", intent_recognition_node)
    graph.add_node("router", router_node)
    graph.add_node("search", search_tool_handler)
    graph.add_node("summarize", summarize_handler)
    graph.add_node("sentiment", sentiment_handler)
    graph.add_node("fallback", fallback_handler)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("intent_recognition")
    graph.add_edge("intent_recognition", "router")

    graph.add_conditional_edges(
        "router",
        route_next,
        {
            "search": "search",
            "summarize": "summarize",
            "sentiment": "sentiment",
            "fallback": "fallback",
            "finalize": "finalize",
        },
    )

    for node_name in ("search", "summarize", "sentiment", "fallback"):
        graph.add_edge(node_name, "router")

    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info("Граф агента успешно собран.")
    return compiled