"""The real LLM providers, exercised against a stubbed transport.

These paths cannot be tested against the live APIs — there is no key here, and a test suite
should not depend on a third party being up or burn a student's free quota. But they *will* run
the moment someone sets `GROQ_API_KEY`, so shipping them unexercised means the first real
request is also the first test.

What is stubbed is only the wire. Request construction, response parsing, tool-call decoding and
error handling are the real code.
"""

from __future__ import annotations

import json

import pytest

from core.agent.providers import (
    EchoProvider,
    GeminiProvider,
    GroqProvider,
    Message,
    MockProvider,
    ToolCall,
    get_provider,
    resolve_provider_name,
)
from core.types import AgentTool

TOOL = AgentTool(
    name="get_churn_model_metrics",
    description="Report churn model metrics.",
    module_id="churn",
    parameters_schema={"type": "object", "properties": {}},
)


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def stub_post(monkeypatch, response: FakeResponse, capture: dict):
    """Replace httpx.post in the providers module and record what it was called with."""
    import core.agent.providers as mod

    def fake_post(url, **kwargs):
        capture["url"] = url
        capture["json"] = kwargs.get("json")
        capture["headers"] = kwargs.get("headers")
        capture["params"] = kwargs.get("params")
        return response

    monkeypatch.setattr(mod.httpx, "post", fake_post)


# --- Groq --------------------------------------------------------------------


def test_groq_requires_a_key() -> None:
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqProvider(api_key="")


def test_groq_decodes_tool_calls(monkeypatch) -> None:
    capture: dict = {}
    stub_post(
        monkeypatch,
        FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "function": {
                                        "name": "get_churn_model_metrics",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        ),
        capture,
    )

    out = GroqProvider(api_key="gsk-test").complete([Message(role="user", content="hi")], [TOOL])

    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "get_churn_model_metrics"
    assert out.tool_calls[0].id == "call_abc"
    assert out.tool_calls[0].arguments == {}
    # Tool binding must actually be sent, or the model can never call anything.
    assert capture["json"]["tools"], "tools were not passed to Groq"
    assert capture["json"]["tool_choice"] == "auto"
    assert capture["headers"]["Authorization"] == "Bearer gsk-test"


def test_groq_survives_malformed_tool_arguments(monkeypatch) -> None:
    """A model that emits invalid JSON must not crash the agent loop."""
    capture: dict = {}
    stub_post(
        monkeypatch,
        FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c1", "function": {"name": "t", "arguments": "{not json"}}
                            ]
                        }
                    }
                ]
            },
        ),
        capture,
    )
    out = GroqProvider(api_key="gsk-test").complete([Message(role="user", content="hi")], [TOOL])
    assert out.tool_calls[0].arguments == {}


def test_groq_raises_a_readable_error_on_4xx(monkeypatch) -> None:
    capture: dict = {}
    stub_post(monkeypatch, FakeResponse(401, text="invalid api key"), capture)
    with pytest.raises(RuntimeError, match="Groq 401"):
        GroqProvider(api_key="gsk-bad").complete([Message(role="user", content="hi")], [TOOL])


def test_groq_omits_tools_when_none_are_bound(monkeypatch) -> None:
    """Sending an empty tools array makes some providers error; omit the key entirely."""
    capture: dict = {}
    stub_post(monkeypatch, FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}), capture)
    GroqProvider(api_key="gsk-test").complete([Message(role="user", content="hi")], [])
    assert "tools" not in capture["json"]


# --- Gemini -------------------------------------------------------------------


def test_gemini_requires_a_key() -> None:
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="")


def test_gemini_declares_tools_in_its_own_shape(monkeypatch) -> None:
    """Gemini uses tools[].functionDeclarations, not OpenAI's tools[].function."""
    capture: dict = {}
    stub_post(
        monkeypatch,
        FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}),
        capture,
    )
    GeminiProvider(api_key="ai-test").complete([Message(role="user", content="hi")], [TOOL])

    decls = capture["json"]["tools"][0]["functionDeclarations"]
    assert decls[0]["name"] == "get_churn_model_metrics"
    # The key travels as a query param on Gemini, not an Authorization header.
    assert capture["params"] == {"key": "ai-test"}
    assert capture["url"].endswith(":generateContent")


def test_gemini_decodes_function_calls(monkeypatch) -> None:
    capture: dict = {}
    stub_post(
        monkeypatch,
        FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_churn_model_metrics",
                                        "args": {"k": 3},
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        ),
        capture,
    )
    out = GeminiProvider(api_key="ai-test").complete([Message(role="user", content="hi")], [TOOL])
    assert out.tool_calls[0].name == "get_churn_model_metrics"
    assert out.tool_calls[0].arguments == {"k": 3}


def test_gemini_separates_system_from_contents(monkeypatch) -> None:
    """Gemini takes systemInstruction separately; putting it in contents degrades output."""
    capture: dict = {}
    stub_post(
        monkeypatch,
        FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}),
        capture,
    )
    GeminiProvider(api_key="ai-test").complete(
        [Message(role="system", content="You are Saros."), Message(role="user", content="hi")], []
    )
    assert capture["json"]["systemInstruction"]["parts"][0]["text"] == "You are Saros."
    assert all(c.get("role") != "system" for c in capture["json"]["contents"])


def test_gemini_handles_an_empty_candidates_array(monkeypatch) -> None:
    """A safety block returns no candidates; that must not raise a KeyError."""
    capture: dict = {}
    stub_post(monkeypatch, FakeResponse(200, {"candidates": []}), capture)
    out = GeminiProvider(api_key="ai-test").complete([Message(role="user", content="hi")], [])
    assert out.content == ""
    assert out.tool_calls == []


# --- provider selection -------------------------------------------------------


def test_auto_picks_groq_when_only_that_key_is_set(monkeypatch) -> None:
    """The default must be 'use the key that is present', not 'mock'.

    The deploy this guards against had GROQ_API_KEY set on Render while SAROS_LLM stayed
    unset, so the integrations panel said Groq LIVE and the chat said MOCK. Both readings
    were true and the product looked broken.
    """
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(get_provider(), GroqProvider)


def test_auto_prefers_groq_over_gemini_when_both_keys_exist(monkeypatch) -> None:
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    assert isinstance(get_provider(), GroqProvider)


def test_auto_picks_gemini_when_only_gemini_key_is_set(monkeypatch) -> None:
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    assert isinstance(get_provider(), GeminiProvider)


def test_auto_falls_back_to_mock_with_no_keys(monkeypatch) -> None:
    """Zero keys is the free-tier case and must stay a working product, not a 500."""
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(get_provider(), MockProvider)


def test_explicit_saros_llm_overrides_a_present_key(monkeypatch) -> None:
    """Deliberate mock demos must stay possible; auto must not overrule an explicit choice."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("SAROS_LLM", "mock")
    assert isinstance(get_provider(), MockProvider)
    monkeypatch.setenv("SAROS_LLM", "echo")
    assert isinstance(get_provider(), EchoProvider)


def test_auto_resolves_the_same_name_without_constructing_the_provider(monkeypatch) -> None:
    """`resolve_provider_name` is what the badge and /api/v1/agent report.

    It has to agree with `get_provider`, or the UI can advertise a provider that is not the
    one answering — the exact bug this whole change exists to remove.
    """
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert resolve_provider_name() == "groq" == get_provider().name

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    assert resolve_provider_name() == "gemini" == get_provider().name

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert resolve_provider_name() == "mock" == get_provider().name


def test_an_unknown_provider_name_still_uses_the_auto_rule(monkeypatch) -> None:
    """A typo in SAROS_LLM is not a request for the mock when a key is configured."""
    monkeypatch.setenv("SAROS_LLM", "openai")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert resolve_provider_name() == "groq"


def test_missing_key_falls_back_to_mock_not_an_exception(monkeypatch) -> None:
    """The single most important deployment behaviour: no key must still yield a working hub."""
    monkeypatch.setenv("SAROS_LLM", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert isinstance(get_provider(), MockProvider)


def test_key_present_selects_the_real_provider(monkeypatch) -> None:
    monkeypatch.setenv("SAROS_LLM", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert isinstance(get_provider(), GroqProvider)

    monkeypatch.setenv("SAROS_LLM", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    assert isinstance(get_provider(), GeminiProvider)


def test_unknown_provider_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.setenv("SAROS_LLM", "openai")
    assert isinstance(get_provider(), MockProvider)


def test_echo_provider_isolates_graph_bugs() -> None:
    """Used to tell "the loop is broken" apart from "the model is unhelpful"."""
    out = EchoProvider().complete([Message(role="user", content="ping")], [])
    assert out.content == "[echo] ping"


def test_tool_call_round_trips_through_the_graph(registry, monkeypatch) -> None:
    """A stubbed real provider must drive the actual agent loop, not just parse in isolation."""
    from core.agent.graph import HubAgent

    capture: dict = {}
    calls = {"n": 0}
    import core.agent.providers as mod

    def fake_post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {
                                            "name": "get_churn_model_metrics",
                                            "arguments": "{}",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        return FakeResponse(200, {"choices": [{"message": {"content": "PR-AUC is 0.677."}}]})

    monkeypatch.setattr(mod.httpx, "post", fake_post)

    agent = HubAgent(registry, GroqProvider(api_key="gsk-test"))
    out = agent.run("What are the churn model metrics?")
    assert out["tools_used"] == ["get_churn_model_metrics"], out
    assert out["answer"] == "PR-AUC is 0.677."
