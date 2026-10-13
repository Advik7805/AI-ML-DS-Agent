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
import mimetypes
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.agent.graph import HubAgent
from core.agent.providers import Message, get_provider
from core.contract import ModuleRegistry
from core.integrations.ratelimit import get_limiter
from core.types import (
    DatasetCard,
    Explanation,
    MetricsReport,
    ModuleManifest,
    Prediction,
    UploadAnalysis,
)

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Discover the bodies, build the agent, and start warming models in the background.

    Warm-up is a daemon thread and is *not* awaited: a body trains lazily on first use, so
    without this the first user request pays for training on top of the cold start — a 14.5 s
    tool call in a real trace. `/health` must stay instant, and a health check that blocks on
    model training is a health check that fails the deploy.
    """
    global agent
    registry.discover("modules")
    agent = HubAgent(registry, get_provider())
    from core import warmup

    warmup.start_background(registry)
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
    #: Optional: makes session-uploaded documents visible to knowledge tools.
    session_id: str | None = None


class ToolCallResponse(BaseModel):
    tool: str
    module_id: str
    result: Any


# --- platform ----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Always instant. Reports warm-up *state*, never waits for it."""
    from core import warmup

    warm = warmup.state()
    return {
        "status": "ok",
        "modules_loaded": registry.ids(),
        "load_errors": getattr(registry, "load_errors", {}),
        "warmup": {
            "status": warm["status"],
            "seconds": warm["seconds"],
            "ready": sum(1 for b in warm["bodies"].values() if b.get("ok")),
            "total": len(warm["bodies"]) or len(warmup.trainable_bodies()),
        },
    }


@app.get("/api/v1/warmup")
def warmup_state() -> dict[str, Any]:
    """Per-body warm-up outcome: which models are loaded, which failed, and how long it took.

    Exposed because the alternative is guessing: if the first tool call is still slow, this says
    whether a model is still training or whether something actually failed.
    """
    from core import warmup

    return {"state": warmup.state(), "trainable": list(warmup.trainable_bodies())}


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


@app.get("/api/v1/modules/{module_id}/dataset", response_model=DatasetCard)
def module_dataset(module_id: str) -> DatasetCard:
    """Where this body's numbers come from: company, unit, file, rows, key, caveats.

    Every body answers this. A model that cannot say what it was trained on has no business
    making a decision, so the provenance is an endpoint rather than a paragraph in a README.
    """
    try:
        return registry.get(module_id).dataset_card()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@app.get("/api/v1/tenant")
def tenant() -> dict[str, Any]:
    """The single company every body belongs to, plus each body's dataset card.

    This is the "so what is this, actually?" answer: one page naming the tenant, its business
    units, and which dataset sits behind each body — including which of them are real public
    datasets and which are generated, because those are different claims.
    """
    from core.tenant import TENANT

    return {
        "tenant": TENANT,
        "bodies": [
            {"id": m.id, "dataset": m.dataset_card().model_dump()} for m in registry.all()
        ],
    }


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


# --- analyze your own data ------------------------------------------------------
#
# One endpoint for every body. The multipart file is read into memory under a hard cap, scored,
# and dropped: nothing here touches the filesystem, which is why a stranger's export cannot end
# up in the next visitor's knowledge base.


async def _read_capped(file: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes. The cap is enforced *while* reading, not after.

    A 10 MB check that only runs once the whole body is in memory does not protect a 512 MB
    instance from a 2 GB upload.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail={
                    "message": (
                        f"'{file.filename}' is larger than the {limit // (1024 * 1024)} MB cap."
                    ),
                    "max_bytes": limit,
                    "hint": (
                        "This endpoint processes in memory on a free tier. Send a sample of the "
                        "file — a few thousand rows is enough to see how the model behaves — or "
                        "split it and upload the parts in sequence."
                    ),
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _upload_session(request: Request, session_id: str | None) -> Any:
    from core.sessions import STORE

    return STORE.get_or_create(session_id or request.headers.get("x-saros-session"))


@app.post("/api/v1/modules/{module_id}/analyze", response_model=UploadAnalysis)
async def analyze_upload(
    module_id: str,
    request: Request,
    file: UploadFile,
    session_id: str | None = Form(default=None),
) -> UploadAnalysis:
    """Score *your* file with this body's model. In memory, capped, never persisted.

    Returns the same ``MetricsReport`` and ``Prediction`` shapes the body's own pages render, so
    a result computed on your data sits next to the shipped-tenant result without a second
    rendering path — and the ``columns_report`` says which of your columns were actually used.
    """
    try:
        module = registry.get(module_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    from core.upload import MAX_FILE_BYTES
    from core.upload import UploadRejected as Rejected

    data = await _read_capped(file, MAX_FILE_BYTES)
    if not data:
        raise HTTPException(status_code=422, detail={"message": "The uploaded file is empty."})

    session = _upload_session(request, session_id)
    filename = (file.filename or "upload").strip()

    try:
        result = module.analyze_upload(filename, data, session)
    except Rejected as exc:
        # A refusal that names the columns is the whole feature. 422 + structured detail.
        raise HTTPException(status_code=422, detail=exc.as_detail()) from None
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from None

    from core.sessions import UploadRecord

    session.add(
        UploadRecord(
            upload_id=f"u_{len(session.uploads) + 1}",
            filename=filename,
            module_id=module_id,
            size_bytes=len(data),
            kind=result.kind,
            rows=result.rows,
            columns=[str(c) for c in result.preview[0].keys()] if result.preview else [],
            summary={
                k: v
                for k, v in {
                    "rows_scored": result.rows,
                    "chunks_indexed": (result.metrics.scores or {}).get("chunks_indexed")
                    if result.metrics
                    else None,
                    "top_score": max((p.score for p in result.predictions), default=None),
                    "expected_columns": ", ".join(result.columns_report.expected[:13]) or None,
                }.items()
                if v is not None
            },
        )
    )
    return result


# --- committed sample data ------------------------------------------------------

#: The checkout root: `apps/api/main.py` -> `apps/api` -> `apps` -> the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sample_index() -> list[dict[str, Any]]:
    """What ``samples/`` holds, described by the script that generates it.

    The generator is imported lazily and defensively. It is a development script, and this is
    the gateway: a deployment that shipped without ``scripts/`` or ``samples/`` must still boot
    and answer every other route, with a sentence where the sample would have been — never a 500
    (the rule ``tests/test_routes.py`` exists to enforce, after a route once returned exactly
    that).
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scripts.make_samples import SAMPLES_LIST
    except Exception:  # noqa: BLE001 - any failure here means "this image ships no samples"
        return []

    out: list[dict[str, Any]] = []
    for sample in SAMPLES_LIST:
        path = PROJECT_ROOT / "samples" / sample.relative
        out.append(
            {
                "module_id": sample.module_id,
                "filename": path.name,
                "path": f"samples/{sample.relative}",
                "what": sample.what,
                "bytes": path.stat().st_size if path.exists() else 0,
                "available": path.exists(),
            }
        )
    return out


@app.get("/api/v1/samples")
def list_samples() -> dict[str, Any]:
    """One sample file per body, so "drop a file" has an answer when you have not got one.

    They are committed, so serving them costs nothing — and `tests/test_samples.py` pushes every
    one of them through its body's own `analyze_upload`, which is what makes "known to be
    accepted" a claim rather than a hope.
    """
    samples = _sample_index()
    if not samples:
        return {
            "samples": [],
            "generator": "python scripts/make_samples.py",
            "note": (
                "This deployment serves no samples (samples/ is not present in the image). Run "
                "`python scripts/make_samples.py` in a checkout, or upload a file of your own."
            ),
        }
    return {
        "samples": samples,
        "generator": "python scripts/make_samples.py",
        "note": (
            "Each file is produced by its body's own loader or generator, so its columns cannot "
            "drift from what that body's model expects. GET /api/v1/samples/{module_id} returns "
            "one as a file; where a body has two (maintenance: telemetry with and without a "
            "header), this list is the index of both."
        ),
    }


@app.get("/api/v1/samples/{module_id}")
def get_sample(module_id: str) -> Response:
    """The committed sample for this body, as a file the studio can upload unchanged.

    Raw bytes with the filename in ``Content-Disposition``, deliberately: the browser wraps them
    in a ``File`` and posts them to the same ``/analyze`` route a dropped file takes. A JSON
    envelope the UI unpacks, or a client-side demo that scores without uploading, would be a
    second code path — and a second code path is a second thing that can be wrong about what the
    body accepts.
    """
    samples = _sample_index()
    if not samples:
        raise HTTPException(
            status_code=404,
            detail=(
                "This deployment serves no samples (samples/ is not present in the image). "
                "Run `python scripts/make_samples.py` in a checkout, or upload your own file."
            ),
        )
    matches = [s for s in samples if s["module_id"] == module_id and s["available"]]
    if not matches:
        known = sorted({s["module_id"] for s in samples if s["available"]})
        raise HTTPException(
            status_code=404,
            detail=f"No sample file for '{module_id}'. Available: {', '.join(known) or 'none'}.",
        )

    sample = matches[0]
    path = PROJECT_ROOT / sample["path"]
    return Response(
        content=path.read_bytes(),
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}"',
            # Which committed file this actually is — for the tests, and for anyone reading a
            # network tab and wondering what the button just downloaded.
            "X-Saros-Sample": sample["path"],
        },
    )


@app.get("/api/v1/sessions/{session_id}")
def session_state(session_id: str) -> dict[str, Any]:
    """What this session has uploaded, so the UI can show it after a reload.

    In-memory and TTL'd: an expired id returns ``exists: false`` with the limits, which is
    clearer than a 404 that looks like an error.
    """
    from core.sessions import STORE

    session = STORE.get(session_id)
    if session is None:
        return {
            "exists": False,
            "session_id": session_id,
            "uploads": [],
            "limits": STORE.stats(),
            "note": "No session with this id in this process. Uploads are in-memory and expire.",
        }
    return {
        "exists": True,
        "session_id": session.id,
        "uploads": [u.as_dict() for u in session.uploads],
        "rag_overlay_chunks": len(session.rag_passages),
        "limits": STORE.stats(),
    }


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
    except Exception as exc:  # noqa: BLE001 - a tool must fail with a reason, never an empty body
        # An unhandled exception here used to surface as `500 {"detail": ""}`, which tells the
        # caller nothing. Report what broke and where.
        raise HTTPException(
            status_code=500,
            detail=f"{tool_name} failed: {type(exc).__name__}: {exc}",
        ) from None

    # A document uploaded this session must be visible to the knowledge tool, or "upload a
    # policy and ask about it" would silently only work through the chat path.
    if body.session_id and tool_name == "search_knowledge" and isinstance(result, dict):
        from core.sessions import STORE
        from core.upload import overlay_search

        session = STORE.get(body.session_id)
        if session is not None and session.rag_passages:
            overlay = overlay_search(session, str(body.arguments.get("query", "")), k=3)
            result["session_overlay"] = overlay
            result["session_overlay_note"] = (
                "Passages from documents uploaded in this session, ranked in a separate TF-IDF "
                "space from the shipped corpus — so the two score lists are reported separately "
                "rather than blended into one ranking."
            )
    return ToolCallResponse(tool=tool_name, module_id=owner.id, result=result)


# --- hub agent ---------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    #: Prior turns, oldest first. Kept client-supplied so the API stays stateless on a free
    #: tier, where a sleeping container would otherwise lose every conversation.
    history: list[dict[str, str]] = Field(default_factory=list)
    #: The browser session whose uploads should be visible to the agent. Optional: without it
    #: the agent answers over the shipped tenant data exactly as before.
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    provider: str
    tools_used: list[str]
    trace: list[dict[str, Any]]
    error: str | None = None


def _session_context(session_id: str | None, question: str) -> str | None:
    """Build the agent's context from this session's uploads — including per-question retrieval.

    Two parts, because they answer different questions:

    * the *profile* — which files were uploaded, how many rows, which columns, what the body
      found. Without it the agent cannot talk about "my file" at all.
    * the *retrieved passages* — for documents uploaded to the Knowledge body, the chunks that
      match *this* question, retrieved now rather than pasted in whole. That is what makes a
      document upload a working RAG rather than a file attachment.
    """
    if not session_id:
        return None
    from core.sessions import STORE
    from core.upload import overlay_search

    session = STORE.get(session_id)
    if session is None or (not session.uploads and not session.rag_passages):
        return None

    parts = [session.context_block()]
    if session.rag_passages:
        hits = overlay_search(session, question, k=4)
        if hits:
            lines = [
                "PASSAGES RETRIEVED FROM THE USER'S UPLOADED DOCUMENTS for this question. If you "
                "use them, cite the filename exactly as given.",
            ]
            for h in hits:
                lines.append(f"- `{h['source']}` → {h['heading']} (relevance {h['score']}): {h['snippet']}")
            parts.append("\n".join(lines))
    return "\n\n".join(p for p in parts if p)


def _coerce_history(raw: list[dict[str, str]]) -> list[Message]:
    out: list[Message] = []
    for turn in raw[-12:]:  # bound the context; free tiers have small limits
        role = turn.get("role")
        if role in ("user", "assistant") and turn.get("content"):
            out.append(Message(role=role, content=turn["content"]))  # type: ignore[arg-type]
    return out


@app.get("/api/v1/agent")
def agent_info() -> dict[str, Any]:
    """What is *actually* answering right now — three states, not two.

    ``live: true`` while a templated answer is being served is a lie the badge tells on the
    user's behalf, and it is the same class of lie as the old "Groq is live" badge next to a
    mock chat. A provider that has degraded mid-conversation reports ``degraded: true`` with the
    readable reason, so the UI can say "templated, and here is why" instead of claiming to be a
    language model.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    from core.agent.providers import readable_provider_error, resolve_provider_name

    degraded = bool(getattr(agent.provider, "degraded", False))
    live = agent.provider.name != "mock" and not degraded

    if live:
        note = None
    elif degraded:
        # The provider was reachable at some point and failed. Report what the vendor said, in
        # the same wording the answer's notice uses, so the two never tell different stories.
        reason = readable_provider_error(getattr(agent.provider, "degraded_reason", "") or "")
        note = (
            f"{agent.provider.name} failed, so answers are templated from real tool output until "
            f"it recovers — {reason or 'see the trace for the raw error'}. The next message "
            "retries the live provider automatically."
        )
    else:
        note = (
            "Deterministic mock provider — no LLM key is configured, so answers are templated "
            "from real tool output. Set GROQ_API_KEY or GEMINI_API_KEY (SAROS_LLM is 'auto' by "
            f"default, currently '{resolve_provider_name()}') and the hub switches to a real "
            "model with no code change."
        )

    return {
        "provider": agent.provider.name,
        "provider_requested": os.getenv("SAROS_LLM", "auto"),
        # The badge reads `live`; `degraded` and `provider_model` are what let it tell a mock
        # apart from a live provider that is currently failing.
        "live": live,
        "degraded": degraded,
        "provider_model": getattr(agent.provider, "model", None),
        "max_iterations": agent.max_iterations,
        "tools": [t.name for t in agent.registry.all_tools()],
        "note": note,
    }



@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty.")
    result = agent.run(
        body.message,
        history=_coerce_history(body.history),
        context=_session_context(body.session_id, body.message),
    )
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
        for update in agent.stream(
            body.message,
            history=_coerce_history(body.history),
            context=_session_context(body.session_id, body.message),
        ):
            yield {"event": update.get("node", "update"), "data": json.dumps(update, default=str)}

    return EventSourceResponse(events())
