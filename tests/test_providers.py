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
    GEMINI_MODELS,
    GROQ_MODELS,
    EchoProvider,
    FallbackProvider,
    GeminiProvider,
    GroqProvider,
    Message,
    MockProvider,
    ToolCall,
    _model_is_gone,
    get_provider,
    readable_provider_error,
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


def stub_post_sequence(monkeypatch, responses: list[FakeResponse], capture: dict):
    """Serve one response per call and record every request, for the retry paths.

    Rotation is only observable across calls, so the single-response stub cannot test it: what
    matters is *which* model the second request names.
    """
    import core.agent.providers as mod

    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, "json": kwargs.get("json"), "params": kwargs.get("params")})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    capture["calls"] = calls


def groq_ok(content: str = "answered") -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def gemini_ok(content: str = "answered") -> FakeResponse:
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": content}]}}]})


def model_gone(status: int = 404, model: str = "llama-3.3-70b-versatile") -> FakeResponse:
    """The exact body Groq returned when it retired the old default, seen in the screenshot."""
    return FakeResponse(
        status,
        text=json.dumps(
            {
                "error": {
                    "message": (
                        f"The model {model} does not exist or you do not have access to it."
                    ),
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        ),
    )


def primary_of(provider):
    """The provider that is actually being called, unwrapping the loud fallback wrapper.

    `get_provider()` returns `FallbackProvider(GroqProvider())` for a live provider, so an
    isinstance check against the concrete class has to look through the wrapper — and looking
    through it is worth a helper, because "which provider is answering?" is the question every
    test in this section is really asking.
    """
    return getattr(provider, "primary", provider)


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


# --- model retirement and the candidate list ----------------------------------


def test_a_retired_model_rotates_instead_of_reaching_the_user(monkeypatch) -> None:
    """The bug this whole change exists for, in one test.

    ``llama-3.3-70b-versatile`` was retired on 2026-08-16. The hub had it hardcoded, so Groq
    answered 404, the provider raised, and the raw envelope was rendered where the answer
    belongs. Now the 404 rotates to the next candidate *in the same request*, and the model that
    worked is remembered so the next request starts there rather than paying for the 404 again.
    """
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    capture: dict = {}
    stub_post_sequence(monkeypatch, [model_gone(), groq_ok("recovered")], capture)

    provider = GroqProvider(api_key="gsk-test")
    first = provider.complete([Message(role="user", content="hi")], [])

    assert first.content == "recovered"
    assert [c["json"]["model"] for c in capture["calls"]] == list(GROQ_MODELS[:2])
    assert provider.model == GROQ_MODELS[1], "the working model must be remembered"

    # A second turn starts at the model known to work: one wasted request per deploy, not per turn.
    capture.clear()
    stub_post_sequence(monkeypatch, [groq_ok("again")], capture)
    provider.complete([Message(role="user", content="hi")], [])
    assert [c["json"]["model"] for c in capture["calls"]] == [GROQ_MODELS[1]]


def test_a_bad_key_does_not_burn_every_candidate(monkeypatch) -> None:
    """A 401 is not fixed by another model, and retrying it costs three requests of a free tier.

    Only a failure that says the *model* is gone rotates.
    """
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    capture: dict = {}
    stub_post_sequence(monkeypatch, [FakeResponse(401, text="invalid api key")], capture)

    with pytest.raises(RuntimeError, match="Groq 401"):
        GroqProvider(api_key="gsk-bad").complete([Message(role="user", content="hi")], [])

    assert len(capture["calls"]) == 1, "a rejected key must not be retried against other models"


def test_gemini_rotates_when_its_default_was_shut_down(monkeypatch) -> None:
    """``gemini-2.0-flash`` was shut down on 2026-06-01 — same failure, different vendor."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    capture: dict = {}
    stub_post_sequence(monkeypatch, [model_gone(model="gemini-2.0-flash"), gemini_ok("ok")], capture)

    out = GeminiProvider(api_key="ai-test").complete([Message(role="user", content="hi")], [])

    assert out.content == "ok"
    assert GEMINI_MODELS[0] in capture["calls"][0]["url"]
    assert GEMINI_MODELS[1] in capture["calls"][1]["url"]


def test_a_pinned_model_is_tried_first_but_the_defaults_remain(monkeypatch) -> None:
    """The candidate list is the deploy's choice, then the defaults — both, not either.

    The retired IDs must be gone: a default that has been shut down is a default that produces
    the exact error this file is about, so it cannot be silently kept as a fallback.
    """
    monkeypatch.setenv("GROQ_MODEL", "my-pinned-model")
    monkeypatch.setenv("GEMINI_MODEL", "my-pinned-gemini")
    groq = GroqProvider(api_key="gsk-test")
    gemini = GeminiProvider(api_key="ai-test")

    assert groq.models[0] == "my-pinned-model"
    assert groq.models[1:] == GROQ_MODELS
    assert gemini.models[0] == "my-pinned-gemini"
    assert gemini.models[1:] == GEMINI_MODELS
    for gone in ("llama-3.3-70b-versatile", "gemini-2.0-flash"):
        assert gone not in groq.models and gone not in gemini.models, (
            f"{gone} is retired; leaving it in the candidate list keeps the 404 alive"
        )
    # An explicit `models=` argument is the whole policy, not a prefix to ours.
    assert GroqProvider(api_key="gsk-test", models=("only-this",)).models == ("only-this",)


def test_the_failure_helpers_distinguish_retirement_from_everything_else() -> None:
    """Rotation is only correct for a retirement, and a reader only needs the vendor's sentence.

    ``_model_is_gone`` decides the first; ``readable_provider_error`` decides the second, and it
    is what the API's note and the browser's ``readAgentError`` both build on.
    """
    assert _model_is_gone(404, "not found")
    assert _model_is_gone(400, '{"code":"model_not_found"}')
    assert _model_is_gone(400, "this model has been decommissioned")
    # 400 for any other reason is a malformed request: re-sending it to three models is pointless.
    assert not _model_is_gone(400, "temperature must be <= 2")
    assert not _model_is_gone(401, "invalid api key")
    assert not _model_is_gone(429, "rate limit exceeded")

    raw = (
        'RuntimeError: Groq 404: {"error":{"message":"The model llama-3.3-70b-versatile does '
        'not exist or you do not have access to it.","type":"invalid_request_error"},'
        '"x":1}'
    )
    readable = readable_provider_error(raw)
    assert readable == (
        "Groq 404: The model llama-3.3-70b-versatile does not exist or you do not have access "
        "to it."
    )
    assert "{" not in readable and "RuntimeError" not in readable
    assert readable_provider_error("") == ""
    long = readable_provider_error("RuntimeError: Groq 500: " + "x" * 500)
    assert len(long) <= 180 and long.endswith("…")


def test_the_fallback_provider_answers_from_the_backup_and_says_so(monkeypatch) -> None:
    """A vendor outage must cost the user a sentence, not the answer.

    Three things are asserted together because they are one behaviour: the turn still answers,
    the answer admits what it is, and the advice matches the actual failure — sending someone
    whose model was retired to "check your key" is a support message that cannot work.
    """
    primary = GroqProvider(api_key="gsk-test", models=("gone-model",))
    capture: dict = {}
    stub_post_sequence(monkeypatch, [model_gone(model="gone-model")], capture)
    provider = FallbackProvider(primary, backup=MockProvider())

    message = provider.complete([Message(role="user", content="What does the refund policy say?")], [])

    assert provider.degraded is True
    assert provider.name == "groq" and provider.model == "gone-model"
    assert "does not exist" in (provider.degraded_reason or "")
    # The notice is *in the answer itself*. A degraded reply that only mentions the failure in a
    # server log is how "answer" and "the truth about the answer" drift apart.
    assert message.content.strip()
    assert provider.notice() in message.content
    assert "GROQ_MODEL" in message.content, "advice must name the variable that fixes a retirement"
    assert provider.notice().count("⚠") == 1 and "\n" not in provider.notice()


def test_the_fallback_retries_the_primary_on_the_next_turn(monkeypatch) -> None:
    """Within a turn, one failure is enough. Across turns, the live provider gets another go.

    Without the first half a downed vendor costs a request per plan step; without the second, a
    thirty-second blip would need a redeploy to recover from.
    """
    capture: dict = {}
    stub_post_sequence(monkeypatch, [FakeResponse(401, text="invalid api key")], capture)
    provider = FallbackProvider(
        GroqProvider(api_key="gsk-bad", models=("m1", "m2")), backup=MockProvider()
    )

    provider.complete([Message(role="user", content="rank the top customers by churn risk")], [])
    provider.complete([Message(role="user", content="and how risky is the worst one?")], [])
    assert len(capture["calls"]) == 1, "the rest of the turn must stay on the backup"

    # A rejected key is not a retirement, so the advice must not point at *_MODEL.
    assert "GROQ_MODEL" not in provider.notice()
    assert "invalid api key" in provider.notice()

    # Next turn: the primary is tried again, and when it works the degradation is left behind.
    capture.clear()
    stub_post_sequence(monkeypatch, [groq_ok("live again")], capture)
    provider.reset_turn()
    out = provider.complete([Message(role="user", content="and the metrics?")], [])
    assert out.content == "live again" and provider.notice() not in out.content
    assert len(capture["calls"]) == 1


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
    assert isinstance(primary_of(get_provider()), GroqProvider)


def test_auto_prefers_groq_over_gemini_when_both_keys_exist(monkeypatch) -> None:
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    assert isinstance(primary_of(get_provider()), GroqProvider)


def test_auto_picks_gemini_when_only_gemini_key_is_set(monkeypatch) -> None:
    monkeypatch.delenv("SAROS_LLM", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    assert isinstance(primary_of(get_provider()), GeminiProvider)


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
    groq = get_provider()
    assert isinstance(groq, FallbackProvider) and isinstance(primary_of(groq), GroqProvider)

    monkeypatch.setenv("SAROS_LLM", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    gemini = get_provider()
    assert isinstance(gemini, FallbackProvider) and isinstance(primary_of(gemini), GeminiProvider)


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
