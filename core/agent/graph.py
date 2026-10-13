"""The hub agent — a LangGraph state machine over the module registry.

    START → plan ⇄ act → END

``plan`` asks the provider what to do next. ``act`` dispatches tool calls through the registry,
so whichever body owns a tool handles it. The loop returns to ``plan`` until the provider
answers without calling a tool, or the iteration guard trips.

The agent holds no module-specific knowledge. It sees a flat list of ``AgentTool`` schemas and
a ``dispatch`` callable — which is why adding a body immediately makes it reachable by the
agent, with no change here.
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.agent.providers import Message, Provider
from core.contract import ModuleRegistry
from core.types import AgentTool

SYSTEM_PROMPT = """You are the Saros hub agent. You coordinate independent specialist bodies —
churn, retrieval, fraud, maintenance, pricing, scraping — each exposed to you as a tool.

Rules:
- Prefer calling a tool over guessing. Numbers must come from a tool result, never from memory.
- You may call several tools in sequence to answer one question.
- Report what the tool returned, including its caveats. Do not round a held-out metric up.
- If a tool errors, say so plainly rather than substituting a plausible-looking number.
"""

#: Hard stop. Without it a provider that keeps re-calling a tool loops forever.
MAX_ITERATIONS = 6


class AgentState(TypedDict):
    messages: Annotated[list[Message], operator.add]
    iterations: int
    trace: Annotated[list[dict[str, Any]], operator.add]
    final: str
    error: str | None
    done: bool


class HubAgent:
    """Wraps the compiled graph. One instance per registry; safe to reuse across requests."""

    def __init__(self, registry: ModuleRegistry, provider: Provider, max_iterations: int = MAX_ITERATIONS):
        self.registry = registry
        self.provider = provider
        self.max_iterations = max_iterations
        self.graph = self._build()

    # --- graph ---------------------------------------------------------------

    def _build(self):
        g: StateGraph = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("act", self._act)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._route, {"act": "act", END: END})
        g.add_edge("act", "plan")
        return g.compile()

    def _route(self, state: AgentState) -> str:
        if state.get("done") or state.get("error"):
            return END
        last = state["messages"][-1] if state["messages"] else None
        if last is not None and last.wants_tool and state["iterations"] < self.max_iterations:
            return "act"
        return END

    # --- nodes ---------------------------------------------------------------

    def _plan(self, state: AgentState) -> dict[str, Any]:
        tools = self.registry.all_tools()
        started = time.perf_counter()
        try:
            reply = self.provider.complete(state["messages"], tools)
        except Exception as exc:  # noqa: BLE001 - surface provider failure, do not crash the graph
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "final": "",
                "done": True,
                "trace": [{
                    "node": "plan",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                }],
            }

        entry: dict[str, Any] = {
            "node": "plan",
            "ok": True,
            "provider": self.provider.name,
            "tool_calls": [tc.name for tc in reply.tool_calls],
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }

        # If the provider produced neither an answer nor a tool call, stop rather than spin.
        if not reply.wants_tool and not reply.content.strip():
            entry["note"] = "provider returned an empty turn"
            return {
                "messages": [reply],
                "final": "The provider returned an empty response. Try rephrasing.",
                "done": True,
                "trace": [entry],
            }

        if reply.wants_tool and state["iterations"] >= self.max_iterations:
            entry["note"] = f"iteration cap {self.max_iterations} reached"
            return {
                "messages": [reply],
                "final": reply.content or (
                    f"Stopped after {self.max_iterations} tool rounds without a final answer."
                ),
                "done": True,
                "trace": [entry],
            }

        return {
            "messages": [reply],
            "final": reply.content if not reply.wants_tool else "",
            "done": not reply.wants_tool,
            "trace": [entry],
        }

    def _act(self, state: AgentState) -> dict[str, Any]:
        assistant = state["messages"][-1]
        tool_messages: list[Message] = []
        trace: list[dict[str, Any]] = []

        for call in assistant.tool_calls:
            started = time.perf_counter()
            try:
                result = self.registry.dispatch(call.name, call.arguments)
                payload = json.dumps(result, default=str)
                ok, err = True, None
            except Exception as exc:  # noqa: BLE001 - a body failing must not kill the conversation
                payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                ok, err = False, f"{type(exc).__name__}: {exc}"

            trace.append({
                "node": "act",
                "tool": call.name,
                "arguments": call.arguments,
                "ok": ok,
                "error": err,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            })
            tool_messages.append(
                Message(role="tool", content=payload, name=call.name, tool_call_id=call.id)
            )

        return {"messages": tool_messages, "iterations": state["iterations"] + 1, "trace": trace}

    # --- entry points --------------------------------------------------------

    def _reset_provider(self) -> None:
        """Give a fallback-wrapped provider a fresh chance at the start of every user turn.

        ``FallbackProvider`` stays on its backup for the remainder of the turn it failed in, so a
        downed vendor costs one wasted request per turn rather than one per plan step. The *next*
        message is a new turn and must retry the live provider: outages end and quotas reset, and
        a process that gave up for good would need a redeploy to recover from a 30-second blip.
        Providers without the hook (mock, echo, a bare GroqProvider in a test) are unaffected.
        """
        reset = getattr(self.provider, "reset_turn", None)
        if callable(reset):
            reset()

    def run(
        self,
        user_input: str,
        history: list[Message] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Run one full turn. Returns the answer plus the trace the UI renders as a timeline.

        ``context`` is session-supplied material — the profile of files the user uploaded, plus
        passages retrieved from them for this question. It goes in as a second system message so
        the agent can answer about a file it has never been trained on, without the context
        leaking into the chat history the browser sends back.
        """
        self._reset_provider()
        messages = [Message(role="system", content=SYSTEM_PROMPT)]
        if context:
            messages.append(Message(role="system", content=context))
        messages.extend(history or [])
        messages.append(Message(role="user", content=user_input))

        final_state: AgentState = self.graph.invoke(
            {"messages": messages, "iterations": 0, "trace": [], "final": "", "error": None, "done": False}
        )

        answer = final_state.get("final") or ""
        if not answer and not final_state.get("error"):
            # Provider's last turn had content but the reducer overwrote `final`.
            for m in reversed(final_state["messages"]):
                if m.role == "assistant" and m.content.strip():
                    answer = m.content
                    break

        return {
            "answer": answer,
            "error": final_state.get("error"),
            "provider": self.provider.name,
            "trace": final_state.get("trace", []),
            "tools_used": [t["tool"] for t in final_state.get("trace", []) if t.get("tool")],
        }

    def stream(
        self,
        user_input: str,
        history: list[Message] | None = None,
        context: str | None = None,
    ):
        """Yield each node's update as it happens, so the UI can show the agent thinking."""
        self._reset_provider()
        messages = [Message(role="system", content=SYSTEM_PROMPT)]
        if context:
            messages.append(Message(role="system", content=context))
        messages.extend(history or [])
        messages.append(Message(role="user", content=user_input))

        for chunk in self.graph.stream(
            {"messages": messages, "iterations": 0, "trace": [], "final": "", "error": None, "done": False},
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                for entry in update.get("trace", []):
                    yield {"node": node, **entry}
                if node == "plan" and update.get("done"):
                    yield {"node": "done", "answer": update.get("final", "")}


def tool_names(registry: ModuleRegistry) -> list[str]:
    return [t.name for t in registry.all_tools()]


__all__ = ["HubAgent", "AgentState", "SYSTEM_PROMPT", "MAX_ITERATIONS", "tool_names", "AgentTool"]
