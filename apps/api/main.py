"""The Nexus API gateway.

One FastAPI app that mounts every body generically. Nothing here knows what churn *is* — it
only knows the contract. Adding a module means adding a directory under ``modules/``; this
file does not change.

Run:  ``uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.agent.graph import HubAgent
from core.agent.providers import Message, get_provider
from core.contract import ModuleRegistry
from core.types import Explanation, MetricsReport, ModuleManifest, Prediction

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Discover every body under ``modules/`` once, at startup, then build the hub agent."""
    global agent
    registry.discover("modules")
    agent = HubAgent(registry, get_provider())
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Nexus",
    version="0.1.0",
    description=(
        "One hub, many independent bodies. Every module implements the same contract, so the "
        "gateway, the agent and the frontend treat them uniformly."
    ),
)

# The frontend runs on Vercel in production and on localhost in dev. Vercel preview hosts are
# dynamic, so allow any origin; this API exposes no authenticated state in v0.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

#: Populated by the lifespan hook. Kept module-level so tests can swap it out.
registry = ModuleRegistry()

#: The hub agent. Provider is resolved from env at startup; falls back to the mock so the hub
#: is a working product even with no API key configured.
agent: HubAgent | None = None


class PredictRequest(BaseModel):
    """Free-form feature bag. Each module validates its own required keys."""

    features: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    tool: str
    module_id: str
    result: Any


# --- platform ----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "modules_loaded": registry.ids(),
        "load_errors": getattr(registry, "load_errors", {}),
    }


@app.get("/api/v1/modules", response_model=list[ModuleManifest])
def list_modules() -> list[ModuleManifest]:
    """Every body the hub can render as a tile."""
    return registry.manifests()


@app.get("/api/v1/modules/{module_id}", response_model=ModuleManifest)
def get_module(module_id: str) -> ModuleManifest:
    try:
        return registry.get(module_id).manifest
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


# --- per-body endpoints (generic — no module-specific code) -------------------


@app.post("/api/v1/modules/{module_id}/predict", response_model=Prediction)
def predict(module_id: str, body: PredictRequest) -> Prediction:
    try:
        module = registry.get(module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    try:
        return module.predict(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None


@app.post("/api/v1/modules/{module_id}/explain", response_model=Explanation)
def explain(module_id: str, body: PredictRequest) -> Explanation:
    try:
        module = registry.get(module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    try:
        return module.explain(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None


@app.get("/api/v1/modules/{module_id}/metrics", response_model=MetricsReport)
def metrics(module_id: str) -> MetricsReport:
    try:
        module = registry.get(module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    try:
        return module.evaluate()
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None


@app.post("/api/v1/modules/{module_id}/train", response_model=MetricsReport)
def train(module_id: str, body: ToolCallRequest | None = None) -> MetricsReport:
    try:
        module = registry.get(module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    try:
        return module.train((body.arguments if body else None) or {})
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None


# --- integrations --------------------------------------------------------------


@app.get("/api/v1/integrations")
def integrations() -> dict[str, Any]:
    """Which external services are live, which are mocked, and what a deploy must set.

    Safe to expose publicly: it reports env var *names*, never values. The frontend reads this
    to render "connect Tavily to enable live search" instead of failing silently.
    """
    from core.integrations.config import summary

    return summary()


# --- agent surface ------------------------------------------------------------


@app.get("/api/v1/agent/tools")
def agent_tools() -> dict[str, Any]:
    """Every tool from every body, ready to bind onto an LLM as function-calling tools."""
    return {"tools": [t.model_dump() for t in registry.all_tools()]}


@app.post("/api/v1/agent/tools/{tool_name}", response_model=ToolCallResponse)
def call_tool(tool_name: str, body: ToolCallRequest) -> ToolCallResponse:
    """Route a tool call to whichever body owns it. This is the hub's execution path."""
    owner = next((m for m in registry.all() if any(t.name == tool_name for t in m.tools())), None)
    if owner is None:
        raise HTTPException(
            status_code=404,
            detail=f"No module owns tool '{tool_name}'.",
        )
    try:
        result = owner.call_tool(tool_name, body.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return ToolCallResponse(tool=tool_name, module_id=owner.id, result=result)


# --- hub agent ---------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    #: Prior turns, oldest first. Kept client-supplied so the API stays stateless on a free
    #: tier, where a sleeping container would otherwise lose every conversation.
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    tools_used: list[str]
    trace: list[dict[str, Any]]
    error: str | None = None


def _coerce_history(raw: list[dict[str, str]]) -> list[Message]:
    out: list[Message] = []
    for turn in raw[-12:]:  # bound the context; free tiers have small limits
        role = turn.get("role")
        if role in ("user", "assistant") and turn.get("content"):
            out.append(Message(role=role, content=turn["content"]))  # type: ignore[arg-type]
    return out


@app.get("/api/v1/agent")
def agent_info() -> dict[str, Any]:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    return {
        "provider": agent.provider.name,
        "max_iterations": agent.max_iterations,
        "tools": [t.name for t in agent.registry.all_tools()],
        "note": (
            "Running on the deterministic mock provider. Set NEXUS_LLM=groq or "
            "NEXUS_LLM=gemini with the matching API key for real reasoning."
            if agent.provider.name == "mock"
            else None
        ),
    }


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty.")
    result = agent.run(body.message, history=_coerce_history(body.history))
    return ChatResponse(
        answer=result["answer"],
        provider=result["provider"],
        tools_used=result["tools_used"],
        trace=result["trace"],
        error=result["error"],
    )


@app.post("/api/v1/agent/chat/stream")
async def chat_stream(body: ChatRequest):
    """Server-sent events: one event per node so the UI can show the agent working."""
    from sse_starlette.sse import EventSourceResponse

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    async def events():
        for update in agent.stream(body.message, history=_coerce_history(body.history)):
            yield {"event": update.get("node", "update"), "data": json.dumps(update, default=str)}

    return EventSourceResponse(events())
