"""Contracts the browser depends on, asserted against the frontend source.

Two failures motivated these, and neither is visible from the API side:

1. **A page that never hydrates.** Next 16 refuses to serve its dev resources (the HMR socket)
   to an origin it does not recognise, and when that socket is refused the dev client never
   hydrates: the page renders beautifully and every button, chip and file picker does nothing,
   with no error anywhere. That is what a preview host saw. The fix is `allowedDevOrigins`, and
   losing it again would silently bring back "none of the controls work".

2. **A session that only half exists.** An upload lives in a server-side session keyed by an id
   the browser sends. When the upload studio and the agent chat minted their own ids, asking the
   agent about the file you just uploaded answered as if it did not exist. One key, one module.

These are source-level assertions on purpose: they are the only way to catch a regression in a
build-time config or in cross-component wiring without a browser in the loop.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "apps" / "web"


def read(relative: str) -> str:
    path = WEB / relative
    assert path.exists(), f"{relative} is missing from the frontend"
    return path.read_text()


# --- 1. the dev server must accept the hosts it is actually run behind ----------------------


def test_dev_origins_are_allowed_or_the_ui_dies_silently() -> None:
    config = read("next.config.ts")
    assert "allowedDevOrigins" in config, (
        "next.config.ts must set allowedDevOrigins: without it a preview host is treated as "
        "cross-origin, the HMR socket is refused, and the app never hydrates"
    )
    match = re.search(r"const DEV_ORIGINS = \[(.*?)\];", config, re.S)
    assert match, "DEV_ORIGINS list not found"
    origins = match.group(1)
    for host in ("localhost", "127.0.0.1", "*.e2b.app"):
        assert host in origins, f"'{host}' must be an allowed dev origin (found: {origins!r})"
    assert "SAROS_DEV_ORIGINS" in origins, "the list should be extendable without editing code"


def test_the_browser_never_calls_the_api_origin_directly() -> None:
    """The preview works because everything is same-origin; a hardcoded :8000 would break it.

    Server components may talk to the API directly — that is what `serverFetch` does on the Next
    server. Anything that runs in the browser must use a relative path.
    """
    offenders = []
    for path in (WEB / "components").rglob("*.tsx"):
        for match in re.finditer(r"https?://(?:localhost|127\.0\.0\.1):?\d*", path.read_text()):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert not offenders, "components must use relative /api paths:\n  " + "\n  ".join(offenders)

    # The single hardcoded origin in the app is the server-side fallback, and it is documented as
    # such; the browser path is the relative one next to it.
    api = read("lib/api.ts")
    assert re.search(r"API_ORIGIN = .*127\.0\.0\.1:8000", api), (
        "the server-side fallback origin should stay explicit"
    )
    assert "export async function clientFetch" in api and "/api/" in api


# --- 2. one session id, carried by every request that needs it -------------------------------


def test_one_session_key_shared_by_studio_and_chat() -> None:
    session_module = read("lib/session.ts")
    assert "saros.session" in session_module, "the session key belongs in lib/session.ts"
    for consumer in ("components/hub/UploadStudio.tsx", "components/hub/AgentChat.tsx"):
        source = read(consumer)
        assert "@/lib/session" in source, f"{consumer} must import the shared session module"
        assert "localStorage" not in source, (
            f"{consumer} must not touch localStorage itself — that is how the two components "
            "ended up with different ids"
        )


def test_the_chat_request_carries_the_session_id() -> None:
    source = read("components/hub/AgentChat.tsx")
    body = re.search(r'JSON\.stringify\(\{([^}]*)\}\)', source, re.S)
    assert body, "chat request body not found"
    assert "session_id" in body.group(1), (
        "the chat must send session_id, or an uploaded file is invisible to the agent"
    )


def test_sample_questions_are_asked_not_just_pasted() -> None:
    """A sample question that only fills the input leaves the user one click short of an answer."""
    source = read("components/hub/AgentChat.tsx")
    assert "saros:ask" in source
    start = source.index("const onAsk")
    end = source.index('addEventListener("saros:ask"')
    listener = source[start:end]
    assert "setInput(detail)" in listener, "the question should still be shown in the input"
    assert re.search(r"sendRef\.current\(detail\)|send\(detail\)", listener), (
        "the saros:ask handler must send the question, not only prefill it"
    )


def test_the_upload_studio_offers_every_body() -> None:
    """The picker is driven by the manifests, so a body added later appears with no code change."""
    source = read("components/hub/UploadStudio.tsx")
    assert "modules.map" in source, "the body picker must render whatever the API discovered"
    assert "/analyze" in source, "the studio must post to the body's analyze route"


# --- 3. an error is not an answer ------------------------------------------------------------


def without_comments(source: str) -> str:
    """Source with its `//` and `/* */` comments removed.

    The "must not appear" assertions below look at code only: the comment next to the fix quotes
    the old buggy line for the next reader, and a comment describing a bug is documentation
    rather than a regression.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", line) for line in source.splitlines())


def test_the_ui_never_renders_a_raw_provider_error() -> None:
    """A retired model reached a visitor as `RuntimeError: Groq 404: {"error":{...}}`.

    Source-level assertions on purpose. The bug was a *rendering* decision — the old line
    ``setAnswer(payload.answer || payload.detail || payload.error || ...)`` is one token away
    from looking reasonable, and no amount of API testing can see which variable a component
    writes an error into. `readAgentError` is the one place that turns a vendor failure into a
    sentence, and both components must route through it.
    """
    helper = read("lib/agentError.ts")
    assert "export function readAgentError" in helper, "the one readable-error helper is missing"
    assert "Underneath:" in helper, "the vendor's own words must survive as evidence"

    studio = read("components/hub/UploadStudio.tsx")
    assert "@/lib/agentError" in studio, "the upload studio must use the shared error reader"
    assert (
        "setAnswer(payload.answer || payload.detail || payload.error" not in without_comments(studio)
    ), "the answer slot must never be handed a raw error — that is the bug, verbatim"
    assert "setAnswerError(readAgentError(payload.error))" in studio
    assert "{answerError &&" in studio, "the readable reason must be rendered, not just stored"

    chat = read("components/hub/AgentChat.tsx")
    assert "@/lib/agentError" in chat
    assert "readAgentError(res.error) || null" in chat
    assert "That question could not be answered just now." in chat, (
        "an empty answer needs a sentence of its own, not an error string as the bubble"
    )
    assert "The agent is unreachable right now." in chat
