from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


IntentType = Literal["search", "summarize", "sentiment", "none"]


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_query: str
    detected_intents: list[IntentType] = Field(default_factory=list)
    intermediate_result: Optional[str] = None
    final_answer: Optional[str] = None

    next_intent_index: int = 0
    execution_chain: list[IntentType] = Field(default_factory=list)


class IntentRecognitionOutput(BaseModel):
    intents: list[IntentType] = Field(
        default_factory=list,
        description="Интенты из набора search/summarize/sentiment/none",
    )


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResults(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)


class LLMSettings(BaseModel):
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


class NodeModelsConfig(BaseModel):
    models: list[LLMSettings] = Field(default_factory=list)


class LLMConfig(BaseModel):
    defaults: LLMSettings
    nodes: dict[str, NodeModelsConfig] = Field(default_factory=dict)