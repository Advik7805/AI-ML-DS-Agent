"""The Saros API gateway.

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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.agent.graph import HubAgent
from core.agent.providers import Message, get_provider
from core.contract import ModuleRegistry
from core.integrations.ratelimit import get_limiter
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
    title="Saros",
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


#: Paths never metered. `/health` is hit by the keep-alive cron every 10 minutes and must not
#: be the request that trips the limit; `/` is the OpenAPI shell.
_RATE_LIMIT_EXEMPT = frozenset({"/health", "/", "/docs", "/openapi.json", "/redoc"})


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Cap requests per client, so one crawler cannot exhaust a free-tier budget.

    Only the mutating and compute-heavy paths are metered. `/health` is exempt because the
    keep-alive cron hits it every 10 minutes and must never be the thing that trips the limit,
    and the read-only discovery routes are cheap enough not to bother.

    A limiter that raises would be worse than no limiter, so any failure inside it is swallowed
    and the request proceeds. Availability beats accounting here.
    """
    if request.url.path in _RATE_LIMIT_EXEMPT or request.method == "OPTIONS":
        return await call_next(request)

    try:
        limiter = get_limiter()
        # Client IP if a proxy set one, else the socket peer. Unauthenticated, so this is a
        # coarse brake rather than a security boundary — and it says so in the headers.
        client = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if not client and request.client:
            client = request.client.host
        decision = limiter.check(f"{client or 'unknown'}:{request.url.path}")
    except Exception:  # noqa: BLE001 - never let accounting take the API down
        return await call_next(request)

    if not decision.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit of {decision.limit} requests per {decision.reset_in}s exceeded "
                    f"for this client. Counted by {decision.backend}."
                )
            },
            headers=decision.headers(),
        )

    response = await call_next(request)
    for k, v in decision.headers().items():
        response.headers[k] = v
    return response

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


@app.get("/api/v1/runs")
def runs(limit: int = 20, body: str | None = None) -> dict[str, Any]:
    """Experiment tracking: which run produced the model that is currently loaded.

    Seven bodies, each selecting an estimator by a metric, makes "which training run is live"
    unanswerable without a log. This reads it. Backed by MLflow when `MLFLOW_TRACKING_URI` is
    set, otherwise by a local JSONL file — `status.backend` says which, so the answer is never
    a guess.

    Render's free filesystem is ephemeral, so the JSONL log does not survive a redeploy. That is
    stated in the response rather than left to be discovered.
    """
    from core.tracking import recent_runs, status

    st = status()
    if st["backend"] == "jsonl":
        st["warning"] = (
            "Local JSONL log on an ephemeral filesystem — runs are lost on redeploy. Set "
            "MLFLOW_TRACKING_URI to keep history across deploys. " + (st.get("warning") or "")
        ).strip()
    return {"status": st, "runs": recent_runs(limit=limit, body=body)}


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
            "Running on the deterministic mock provider because no supported LLM key is configured "
            "or the explicitly selected provider key is missing. Set GROQ_API_KEY or GEMINI_API_KEY "
            "for automatic live selection, or SAROS_LLM=mock to stay offline."
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
