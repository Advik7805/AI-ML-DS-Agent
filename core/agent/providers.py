"""LLM providers for the hub agent.

One protocol, four implementations:

- ``MockProvider``   — deterministic, no network, no keys. Used by the test suite and by the
  demo so the *whole* agent loop (plan → tool → observe → synthesise) is exercisable offline.
  It calls the same real tools and summarises the same real numbers; only the language model
  is replaced by a policy.
- ``GroqProvider``   — OpenAI-compatible ``/chat/completions``, free tier.
- ``GeminiProvider`` — Google's REST API, free tier.
- ``EchoProvider``   — never calls anything; useful for isolating graph bugs.

Selection is by env var, and the **default is ``auto``**: a real model when a key is present, the
labelled mock when none is. That default matters, because the failure it prevents is invisible.
A deploy that has ``GROQ_API_KEY`` set but leaves ``SAROS_LLM`` unset used to sit on the mock
provider while the integrations panel reported Groq as live — the product looked broken and the
only clue was a badge nobody read.

    SAROS_LLM=auto | groq | gemini | mock | echo      # auto is the default
    GROQ_API_KEY=...
    GEMINI_API_KEY=...

``auto`` resolves in this order: Groq, then Gemini, then mock. An explicit ``SAROS_LLM`` always
wins, including ``SAROS_LLM=mock``, which is how you demo the deterministic path deliberately
rather than by accident. If a key is present but rejected by the vendor, the failure surfaces at
call time as a readable error — it does not silently downgrade to the mock mid-conversation.

Groq and Gemini are called over plain HTTP rather than through their SDKs: two fewer heavy
dependencies in an image that has to fit in 512 MB on a free tier.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from core.types import AgentTool

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class Provider(Protocol):
    """Everything the agent loop needs from a language model."""

    name: str

    def complete(self, messages: list[Message], tools: list[AgentTool]) -> Message:
        """Return the next assistant turn: either tool calls or a final answer."""


# --- mock --------------------------------------------------------------------

#: Keyword → tool, ordered most-specific first.
#:
#: Body-scoped patterns must come before generic ones, or "how good is the fraud model"
#: resolves to the *churn* metrics tool. That bug shipped once already.
_TOOL_HINTS: list[tuple[str, str]] = [
    # Live-web intents sit first: they are the most specific, and a bare word like "churn" or
    # "pricing" further down would otherwise win on "search the web for churn benchmarks".
    # `extract_url` deliberately omits "scrape" and "fetch" — those belong to `scrape_url`.
    (r"(?=[\s\S]*\b(render\w*|browser|javascript|js-rendered|client-rendered|dynamic\w*|firecrawl)\b)"
     r"(?=[\s\S]*https?://)",
     "render_url"),
    (r"(?=[\s\S]*\b(extract\w*|read|get the text|parse|summarise\w* the page)\b)(?=[\s\S]*https?://)",
     "extract_url"),
    (r"(?=[\s\S]*\b(search\w*|look\w*|find|research\w*|latest|news)\b)"
     r"(?=[\s\S]*\b(web|online|internet|google|tavily|news|recently|this week|today)\b)",
     "web_search"),
    # Drift checks, ahead of the body-scoped metrics patterns. They must win on phrasings like
    # "is the fraud model still valid?" which contain both a body noun and a model noun - left
    # lower down, that question resolves to the metrics tool and the drift check is unreachable.
    (r"(?=[\s\S]*\b(fraud\w*|transactions?|anomal\w*|payment\w*|card\w*)\b)"
     r"(?=[\s\S]*\b(drift\w*|drifted|still (?:valid|accurate|reliable)|distribution shift|"
     r"out of distribution|stale\w*|degrad\w*|psi\b)\b)",
     "check_fraud_drift"),
    (r"(?=[\s\S]*\b(maintenance|rul\b|remaining useful life|turbofans?|engines?|fleets?|machines?|"
     r"assets?|sensors?|telemetry)\b)"
     r"(?=[\s\S]*\b(drift\w*|drifted|still (?:valid|accurate|reliable)|distribution shift|"
     r"out of distribution|stale\w*|degrad\w*|psi\b)\b)",
     "check_fleet_drift"),
    # Body-scoped metrics patterns, all as dual lookaheads. Two reasons they are grouped here
    # and written order-independently:
    #   1. They must precede both the generic metrics hint (which defaults to churn) and the
    #      per-body action tools, whose patterns match the same nouns. "What is the MAE of the
    #      RUL model?" otherwise resolves to `predict_rul` on the word "RUL" alone.
    #   2. `a[\s\S]*b` assumes word order. "How good is the maintenance model?" puts the noun
    #      *before* the metric word, so a sequence pattern silently misses it.
    (r"(?=[\s\S]*\b(fraud\w*|transactions?|anomal\w*)\b)"
     r"(?=[\s\S]*\b(metrics?|model cards?|how (?:good|well)|perform\w*|evaluat\w*)\b)",
     "get_fraud_model_metrics"),
    (r"(?=[\s\S]*\b(maintenance|rul\b|remaining useful life|turbofans?|engines?|fleets?|machines?|assets?)\b)"
     r"(?=[\s\S]*\b(metrics?|model cards?|how (?:good|well)|perform\w*|evaluat\w*|mae\b|rmse\b|r2\b|r-?squared)\b)",
     "get_maintenance_model_metrics"),
    (r"(?=[\s\S]*\b(churn\w*|customers?|retention)\b)"
     r"(?=[\s\S]*\b(metrics?|model cards?|how (?:good|well)|pr-?auc|roc(?:-auc)?|thresholds?|accuracy)\b)",
     "get_churn_model_metrics"),
    (r"\b(fairness|bias|audit\w*|disparate|equit\w*|demographic)\b", "audit_model_fairness"),
    (r"\b(diabetes|patients?|clinical|glucose|hba1c|diagnos\w*|health risk)\b", "assess_diabetes_risk"),
    # Dual lookaheads, not `a[\s\S]*b`. "rank engines by maintenance urgency" puts the verb
    # *before* the noun, so an ordered pattern silently misses and the question falls through to
    # rank_at_risk_customers on the bare word "rank" - a churn tool answering a maintenance
    # question, which is worse than no answer.
    (r"(?=[\s\S]*\b(engines?|turbofans?|fleets?|machines?|assets?)\b)"
     r"(?=[\s\S]*\b(rank\w*|worst|urgent\w*|first|top|priorit\w*|which)\b)",
     "rank_engines_by_urgency"),
    (r"\b(rul|remaining useful life|turbofans?|predictive maintenance|engine health|fail\w* soon)\b", "predict_rul"),
    # Lookaheads rather than `a[\s\S]*b`: word order in a question is not fixed, and an
    # ordered pattern silently fails on "which SKUs have the biggest pricing opportunity".
    (r"(?=[\s\S]*\b(skus?|products?|catalog\w*)\b)(?=[\s\S]*\b(opportunit\w*|rank\w*|list\w*|which|all)\b)",
     "list_skus_by_opportunity"),
    (r"\b(prices?|pricing|elasticit\w*|skus?|revenue optimis\w*|revenue optimiz\w*)\b", "optimise_price"),
    (r"\b(rank\w*|top\b|at[- ]risk|portfolios?|campaigns?|target lists?|worst|most likely|highest risk)\b",
     "rank_at_risk_customers"),
    (r"(?=[\s\S]*\b(fraud\w*|transactions?)\b)(?=[\s\S]*\b(streams?|feeds?|live|right now|currently happening)\b)",
     "summarise_fraud_stream"),
    (r"\b(churn\w*|retention|cancell?\w*|attrit\w*|customers?)\b", "score_customer_churn"),
    # `list_knowledge_sources` must precede `search_knowledge`: the search pattern matches the
    # bare word "knowledge", so "list the knowledge sources" was being answered by *searching*
    # for that phrase, finding nothing, and telling the user the corpus was empty. A question
    # about the corpus is not a question to the corpus.
    (r"(?=[\s\S]*\b(list\w*|show|what'?s|how many|which|catalog\w*)\b)"
     r"(?=[\s\S]*\b(sources?|corpus|documents?|indexed)\b)", "list_knowledge_sources"),
    (r"\b(sources?|corpus|what.{0,12}indexed)\b", "list_knowledge_sources"),
    (r"\b(search\w*|knowledge|polic(?:y|ies)|refunds?|sla\b|what does|approv\w*|discount|playbook|privacy)\b",
     "search_knowledge"),
    # Ahead of scrape_url, whose `scrape\w*` also matches "scraper" in "what can the scraper
    # do". Left below it, a question *about* the scraper was answered by using it.
    (r"\b(capabilit\w*|what can (?:you|the scraper|it)|allowlist|live scraping|what does the scraper)\b",
     "scraper_capabilities"),
    (r"\b(scrape\w*|crawl\w*|fetch|urls?|websites?|competitor\w*|pages?)\b", "scrape_url"),
    (r"\b(metrics?|model cards?|pr-?auc|roc(?:-auc)?|thresholds?|evaluat\w*|how (?:good|well))\b",
     "get_churn_model_metrics"),
    (r"\b(fraud\w*|transactions?|suspicious|flag\w*)\b", "score_transaction"),
]


def _pick_tool(user_text: str, available: set[str]) -> str | None:
    """First matching hint wins, so ordering *is* the routing policy.

    URLs are stripped before matching any pattern that does not explicitly require one. A URL's
    path is not an intent: "extract https://example.com/pricing" routed to the pricing tool, and
    "fetch https://news.ycombinator.com" routed to a news search, purely because of the host and
    path text. Patterns that do need a URL declare it by containing `https?://`.
    """
    url_free = re.sub(r"https?://\S+", " ", user_text)
    for pattern, tool in _TOOL_HINTS:
        if tool not in available:
            continue
        haystack = user_text if "https?://" in pattern else url_free
        if re.search(pattern, haystack, re.IGNORECASE):
            return tool
    return None


#: Typical values for the numeric features across bodies, so a tool call from a bare
#: sentence still exercises the real pipeline instead of failing on required arguments.
_FEATURE_DEFAULTS: dict[str, float] = {
    # churn
    "account_length": 120, "number_vmail_messages": 0,
    "total_day_minutes": 180, "total_day_calls": 110,
    "total_eve_minutes": 200, "total_eve_calls": 100,
    "total_night_minutes": 200, "total_night_calls": 100,
    "total_intl_minutes": 10, "total_intl_calls": 4,
    "number_customer_service_calls": 1,
    "international_plan": 0, "voice_mail_plan": 0,
    # fraud
    "amount": 250.0, "hour_of_day": 13, "distance_from_home_km": 5,
    "txn_count_1h": 1, "distinct_merchants_24h": 2, "card_age_days": 900,
    "avg_amount_30d": 60, "is_night": 0, "is_foreign": 0, "pin_used": 1,
    # diagnostics
    "pregnancies": 2, "plasma_glucose": 120, "blood_pressure": 70,
    "triceps_skin_thickness": 30, "insulin": 120, "bmi": 32,
    "diabetes_pedigree": 0.5, "age": 40,
}

#: Non-required arguments with sensible fallbacks.
_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "n": 10, "k": 5, "min_score": 0.0, "min_demand_share": 0.8,
}


def _extract_features(user_text: str) -> dict[str, float]:
    """Pull any numbers the user typed into plausible feature slots."""
    nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", user_text)]
    defaults = {
        "account_length": 120, "number_vmail_messages": 0,
        "total_day_minutes": 180, "total_day_calls": 110,
        "total_eve_minutes": 200, "total_eve_calls": 100,
        "total_night_minutes": 200, "total_night_calls": 100,
        "total_intl_minutes": 10, "total_intl_calls": 4,
        "number_customer_service_calls": 1,
        "international_plan": 0, "voice_mail_plan": 0,
    }
    # "N service calls" is the one people actually say out loud.
    m = re.search(r"(\d+)\s*(?:service\s*)?calls", user_text, re.IGNORECASE)
    if m:
        defaults["number_customer_service_calls"] = float(m.group(1))
    if re.search(r"international plan", user_text, re.IGNORECASE):
        defaults["international_plan"] = 1
    del nums
    return defaults


class MockProvider:
    """A deterministic policy standing in for the language model.

    It performs real tool selection and writes its answer from the *actual* values the tools
    returned, so the demo shows genuine end-to-end plumbing. What it does not have is open-ended
    reasoning — and the answer says so, rather than pretending otherwise.
    """

    name = "mock"

    def complete(self, messages: list[Message], tools: list[AgentTool]) -> Message:
        available = {t.name for t in tools}
        by_name = {t.name: t for t in tools}

        # Turn 1: decide whether a tool is needed.
        if not any(m.role == "tool" for m in messages):
            user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
            tool = _pick_tool(user_text, available)
            if tool is None:
                return Message(
                    role="assistant",
                    content=(
                        "I can score churn risk, rank an at-risk portfolio, report model metrics, "
                        "search the knowledge base, score a transaction for fraud, or scrape a URL. "
                        "Tell me which. (Running on the deterministic mock provider — no LLM key "
                        "is configured. Set GROQ_API_KEY or GEMINI_API_KEY and the hub switches "
                        "to a real model on the next request.)"
                    ),
                )
            args = self._args_for(tool, user_text, by_name[tool])
            return Message(role="assistant", tool_calls=[ToolCall(id="call_1", name=tool, arguments=args)])

        # Turn 2+: summarise the real tool output.
        last_tool = next(m for m in reversed(messages) if m.role == "tool")
        return Message(role="assistant", content=_summarise(last_tool.content))

    @staticmethod
    def _args_for(tool: str, user_text: str, spec: AgentTool) -> dict[str, Any]:
        """Fill a tool's arguments from the schema, not from a per-tool special case.

        The earlier version special-cased two tools and passed nothing for the rest, so
        ``optimise_price`` failed with a missing ``sku`` and ``assess_diabetes_risk`` with
        eight missing clinical features. Deriving from the JSON Schema means a new body works
        with the mock without touching this file.
        """
        props = spec.parameters_schema.get("properties", {}) or {}
        required = spec.parameters_schema.get("required", []) or []
        args: dict[str, Any] = {}

        m = re.search(r"https?://\S+", user_text)
        sku = re.search(r"\b(SKU[-_ ]?\d+)\b", user_text, re.IGNORECASE)
        top_n = re.search(r"top\s+(\d+)", user_text, re.IGNORECASE)
        calls = re.search(r"(\d+)\s*(?:service\s*)?calls", user_text, re.IGNORECASE)
        numbers = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", user_text)]

        for name, schema in props.items():
            if name not in required and name in _OPTIONAL_DEFAULTS:
                args[name] = _OPTIONAL_DEFAULTS[name]
                continue
            if name in _OPTIONAL_DEFAULTS and name not in required:
                continue

            if name == "url":
                args[name] = m.group(0) if m else "https://example.com"
            elif name == "sku":
                args[name] = sku.group(1).replace(" ", "-").upper() if sku else "SKU-000"
            elif name == "query":
                args[name] = user_text.strip()[:200] or "overview"
            elif name == "n":
                args[name] = int(top_n.group(1)) if top_n else 10
            elif name in _FEATURE_DEFAULTS:
                args[name] = _FEATURE_DEFAULTS[name]
            elif name in ("international_plan",) and re.search(r"international plan", user_text, re.IGNORECASE):
                args[name] = 1
            else:
                types = schema.get("type")
                types = types if isinstance(types, list) else [types]
                if "number" in types or "integer" in types:
                    args[name] = 0
                elif "string" in types:
                    args[name] = user_text.strip()[:100] or "n/a"
                elif "array" in types:
                    # Arrays of URLs are the common case (extract_url). Pull every URL out of
                    # the message rather than inventing one, and fall back to a single
                    # placeholder so the tool still exercises its validation path.
                    found = re.findall(r"https?://\S+", user_text)
                    args[name] = found or ["https://example.com"]
                elif "boolean" in types:
                    args[name] = False
                else:
                    args[name] = None

        # Churn-specific phrasing people actually use.
        if calls and "number_customer_service_calls" in props:
            args["number_customer_service_calls"] = float(calls.group(1))
        if re.search(r"international plan", user_text, re.IGNORECASE) and "international_plan" in props:
            args["international_plan"] = 1
        # First bare number in the text is usually the amount.
        if "amount" in props and numbers:
            args["amount"] = numbers[0]

        return {k: v for k, v in args.items() if k in props}


# --- real providers ----------------------------------------------------------


def _openai_tools(tools: list[AgentTool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class GroqProvider:
    """Groq's OpenAI-compatible endpoint. Fast, and the free tier is genuinely usable."""

    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile", timeout: float = 60.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", model)
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

    def complete(self, messages: list[Message], tools: list[AgentTool]) -> Message:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _openai_messages(messages),
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = _openai_tools(tools)
            payload["tool_choice"] = "auto"

        res = httpx.post(
            self.URL,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if res.status_code >= 400:
            raise RuntimeError(f"Groq {res.status_code}: {res.text[:400]}")
        choice = res.json()["choices"][0]["message"]

        calls = []
        for tc in choice.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.get("id", "call_0"), name=tc["function"]["name"], arguments=args))
        return Message(role="assistant", content=choice.get("content") or "", tool_calls=calls)


class GeminiProvider:
    """Google Gemini over REST. Free tier; function-calling via ``tools[].functionDeclarations``."""

    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash", timeout: float = 60.0):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", model)
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

    def complete(self, messages: list[Message], tools: list[AgentTool]) -> Message:
        contents = []
        system = ""
        for m in messages:
            if m.role == "system":
                system = m.content
            elif m.role == "tool":
                contents.append(
                    {"role": "user", "parts": [{"functionResponse": {"name": m.name, "response": {"result": _safe_json(m.content)}}}]}
                )
            elif m.role == "assistant" and m.tool_calls:
                contents.append(
                    {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": tc.name, "args": tc.arguments}} for tc in m.tool_calls
                        ],
                    }
                )
            else:
                contents.append({"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]})

        body: dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": 0.2}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters_schema or {"type": "object", "properties": {}},
                        }
                        for t in tools
                    ]
                }
            ]

        res = httpx.post(
            self.URL.format(model=self.model),
            params={"key": self.api_key},
            json=body,
            timeout=self.timeout,
        )
        if res.status_code >= 400:
            raise RuntimeError(f"Gemini {res.status_code}: {res.text[:400]}")

        parts = (res.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        calls, text = [], []
        for i, p in enumerate(parts):
            if "functionCall" in p:
                fc = p["functionCall"]
                calls.append(ToolCall(id=f"call_{i}", name=fc["name"], arguments=fc.get("args") or {}))
            elif "text" in p:
                text.append(p["text"])
        return Message(role="assistant", content="\n".join(text), tool_calls=calls)


class EchoProvider:
    """Returns a fixed answer. For isolating graph bugs from provider bugs."""

    name = "echo"

    def complete(self, messages: list[Message], tools: list[AgentTool]) -> Message:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return Message(role="assistant", content=f"[echo] {user}")


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# --- summariser used by the mock ---------------------------------------------


def _summarise(tool_output: str) -> str:
    """Turn a real tool result into a readable answer, using only the values actually present."""
    data = _safe_json(tool_output)

    if isinstance(data, dict):
        if "customers" in data:
            rows = data["customers"]
            if not rows:
                return "No customer in the loaded book clears that risk threshold."
            lines = [
                f"Scored {data['n_scored']:,} customers from the {data['data_source']} dataset. "
                f"Here are the {len(rows)} highest risk:"
            ]
            for c in rows[:5]:
                driver = c.get("top_driver") or "n/a"
                lines.append(
                    f"- {c['state']} (row {c['row_index']}): {c['score']*100:.1f}% — {c['label']}, "
                    f"main driver {driver.replace('_', ' ')}."
                )
            lines.append(f"Recommended first action: {rows[0]['decision']}")
            lines.append(
                "_(mock provider: these are the real model outputs, the wording is templated.)_"
            )
            return "\n".join(lines)

        if "scores" in data:
            s = data["scores"]
            lift = s["pr_auc"] / s["baseline_pr_auc"] if s.get("baseline_pr_auc") else 0
            c = data.get("confusion", {})
            if "mae_cycles" in s:
                bands = " · ".join(
                    f"{k.replace('mae_', '').replace('_cycles', '').replace('_', '-')} cycles {v:.1f}"
                    for k, v in s.items() if k.startswith("mae_") and k != "mae_cycles"
                )
                return (
                    f"RUL model over {int(s.get('n_engines', 0))} engines, grouped cross-validation:\n"
                    f"- MAE **{s['mae_cycles']:.1f} cycles** · RMSE {s['rmse_cycles']:.1f} · "
                    f"R² {s['r2']:.3f}\n"
                    f"- by remaining-life band: {bands}\n"
                    "Overall MAE is dominated by the far-future band, which is the band where a "
                    "precise number matters least."
                )
            if "fairness_fnr_gap" in s:
                return (
                    f"Diagnostics model on {data['n_samples']} held-out patients:\n"
                    f"- ROC-AUC {s['roc_auc']:.3f} · recall {s['recall']:.3f} · "
                    f"balanced accuracy {s['balanced_accuracy']:.3f}\n"
                    f"- fairness false-negative gap **{s['fairness_fnr_gap']*100:.1f} points** "
                    f"across age bands\n"
                    "False-negative rate is the audited quantity in screening."
                )
            if "mae_elasticity" in s:
                return (
                    f"Elasticity recovery over {data['n_samples']} SKUs:\n"
                    f"- MAE **{s['mae_elasticity']:.3f}** · bias {s['bias_elasticity']:+.3f} · "
                    f"mean R² {s['mean_r2']:.3f}\n"
                    f"- {s['share_identified']*100:.0f}% of SKUs have an identified elasticity\n"
                    "Ground truth exists here only because the demand data is generated."
                )
            return (
                f"Model on {data['n_samples']:,} held-out records "
                f"({data['positive_rate']*100:.2f}% positive):\n"
                f"- PR-AUC **{s['pr_auc']:.3f}** vs {s['baseline_pr_auc']:.3f} baseline ({lift:.2f}x lift)\n"
                f"- ROC-AUC {s['roc_auc']:.3f} · recall {s['recall']:.3f} · precision {s['precision']:.3f}\n"
                f"- operating threshold {data['threshold']:.2f}; confusion "
                f"tn {c.get('tn')} / fp {c.get('fp')} / fn {c.get('fn')} / tp {c.get('tp')}\n"
                f"PR-AUC is the honest headline at this positive rate, not ROC-AUC."
            )

        if "score" in data and "label" in data and "module_id" in data:
            # Every Prediction carries its module_id — use it rather than assuming churn,
            # which is what produced "Churn probability" for a pricing recommendation.
            mid = data["module_id"]
            meta = data.get("meta", {}) or {}
            pct = f"{data['score'] * 100:.1f}%"

            if mid == "pricing":
                head = (
                    f"{meta.get('sku')}: elasticity {meta.get('elasticity')} "
                    f"(95% CI {meta.get('elasticity_ci')})"
                )
                detail = (
                    f"Recommended **${meta.get('recommended_price')}** within observed range "
                    f"{meta.get('observed_price_range')} → "
                    f"{(meta.get('estimated_revenue_uplift') or 0) * 100:+.1f}% revenue."
                )
            elif mid == "maintenance":
                head = f"Remaining life **{meta.get('rul_cycles')} of {meta.get('rul_cap')} cycles**"
                detail = f"Urgency {pct}."
            elif mid == "scraper":
                head = f"Capture: {meta.get('bytes', 0):,} bytes via the {meta.get('fetcher')} fetcher"
                detail = f"URL {meta.get('url')}."
            elif mid == "xai_health":
                head = f"Diabetes risk **{pct}**"
                detail = "_Clinical decision support only; not a diagnostic device._"
            elif mid == "fraud":
                head = f"Transaction risk **{pct}**"
                detail = (
                    f"Decisive model: {meta.get('decisive_model')} "
                    f"(supervised {meta.get('supervised_score')}, "
                    f"isolation forest {meta.get('unsupervised_score')})."
                )
            elif mid == "rag":
                head = f"Retrieval confidence **{pct}** — {data['label']}"
                detail = f"{len(meta.get('hits', []))} passages available."
            else:
                head = f"Churn probability **{pct}**"
                drivers = ", ".join(f["feature"].replace("_", " ") for f in (data.get("factors") or [])[:3])
                detail = f"Top drivers: {drivers or 'none reported'}."

            return f"{head} — {data['label']}.\n{detail}\nRecommended: {data['decision']}"

        if "n_flagged" in data and "true_positives" in data:
            note = f" ({data['enrichment_note']})" if data.get("enriched") else ""
            top = ", ".join(f"${t['amount']:.0f}@{t['risk']*100:.0f}%" for t in data.get("top_flagged", [])[:3])
            return (
                f"Scored {data['n_scored']} transactions{note}: {data['n_flagged']} flagged — "
                f"{data['true_positives']} true, {data['false_positives']} false, "
                f"{data['missed_fraud']} missed.\nTop flags: {top or 'none'}."
            )

        if "engines" in data and "n_engines" in data:
            rows = data["engines"]
            if not rows:
                return "No engines in the fleet."
            lines = [f"Fleet of {data['n_engines']} engines ({data['data_source']} data). Most urgent:"]
            for e in rows[:5]:
                lines.append(
                    f"- unit {e['unit']} at cycle {e['cycle']}: **{e['predicted_rul']:.0f} cycles** "
                    f"remaining (urgency {e['urgency']:.2f})"
                )
            return "\n".join(lines)

        if "worst_served_group" in data:
            return (
                f"Fairness audit — grouping: {data['grouping_variable']}.\n"
                f"Worst-served group **{data['worst_served_group']}** with a false-negative rate of "
                f"{(data['worst_served_fnr'] or 0)*100:.1f}%; gap across groups "
                f"{(data['fnr_gap'] or 0)*100:.1f} points.\n"
                f"Verdict: {data['verdict']}\n\n{data['metric_rationale']}"
            )

        if "results" in data and "citation" in data:
            rows = data["results"]
            if not rows:
                return (
                    "Nothing in the knowledge base matches that. "
                    "I am not going to answer from memory."
                )
            grounding = data.get("grounding", "Unknown grounding")
            head = f"**{grounding}** ({data.get('matched_terms', 0)} substantive terms matched)"
            if grounding == "Weak grounding":
                head += " — treat the answer as unverified and ask a clarifying question."
            elif grounding == "No grounding":
                return f"**No grounding.** {data.get('decision', '')}"
            lines = [head, ""]
            for r in rows[:4]:
                lines.append(f"- `{r['source']}.md` → **{r['heading']}** ({r['score']:.2f})")
                lines.append(f"  {r['snippet'][:240]}")
            return "\n".join(lines)

        if "n_documents" in data and "sources" in data:
            return (
                f"The knowledge base holds {data['n_documents']} documents across "
                f"{data['n_passages']} passages: " + ", ".join(f"`{s}`" for s in data["sources"]) + "."
            )

        if "top" in data and "n_skus" in data:
            rows = data["top"]
            if not rows:
                return "No SKU has an identified elasticity, so there is nothing to optimise."
            lines = [
                f"Of {data['n_skus']} SKUs, {data['n_skus'] - data.get('n_not_identified', 0)} have an "
                f"identified price elasticity. Biggest revenue opportunities:"
            ]
            for s in rows[:5]:
                lines.append(
                    f"- `{s['sku']}`: elasticity **{s['elasticity']:.2f}**, "
                    f"price to ${s['recommended_price']:,.2f} "
                    f"(+{s['estimated_revenue_uplift'] * 100:.1f}% revenue, r² {s['r2']:.2f})"
                )
            if data.get("n_not_identified"):
                lines.append(
                    f"\n{data['n_not_identified']} SKU(s) have no identified elasticity "
                    f"({', '.join(data['not_identified'][:4])}) — {data.get('note', '')}"
                )
            return "\n".join(lines)

        if "results" in data and "query" in data and "backend" in data:
            if data.get("backend") == "error":
                return (
                    f"Live web search failed: {data.get('error', 'unknown error')}. "
                    "I would rather say so than guess at an answer."
                )
            rows = data.get("results") or []
            if not rows:
                return f"The web search for {data['query']!r} returned nothing."
            # The single most important line in this branch: mock results look exactly like
            # real ones unless the answer says otherwise.
            banner = (
                ""
                if data.get("live")
                else "⚠ **Mock results — Tavily is not configured, so these are placeholders "
                "and must not be cited.**\n\n"
            )
            lines = [f"{banner}Web search for **{data['query']}** ({data['backend']} backend):"]
            if data.get("answer"):
                lines.append(f"\n{data['answer']}")
            for r in rows[:5]:
                lines.append(f"\n- [{r.get('title', 'untitled')}]({r.get('url', '')})")
                if r.get("content"):
                    lines.append(f"  {str(r['content'])[:280]}")
            if data.get("note"):
                lines.append(f"\n_{data['note']}_")
            return "\n".join(lines)

        if "urls" in data and "backend" in data:
            if data.get("backend") == "error":
                return f"Extraction failed: {data.get('error', 'unknown error')}."
            rows = data.get("results") or []
            banner = (
                ""
                if data.get("live")
                else "⚠ **Mock extraction — Tavily is not configured; no page was fetched.**\n\n"
            )
            lines = [f"{banner}Extracted {len(rows)} URL(s):"]
            for r in rows:
                body = " ".join(str(r.get("raw_content", "")).split())[:300]
                lines.append(f"\n- `{r.get('url', '')}`\n  {body}")
            return "\n".join(lines)

        if "fetcher" in data and "allowlist" in data:
            allow = data["allowlist"] or []
            tavily = data.get("tavily") or {}
            lines = [
                f"Scraper runs in **{'live' if data['live'] else 'mock'}** mode via "
                f"`{data['fetcher']}`, capped at {int(data['max_bytes']):,} bytes.\n"
                + (
                    f"Host allowlist: {', '.join(allow)}."
                    if allow
                    else "No hosts are allowlisted, so plain-HTTP fetching is blocked for every URL."
                ),
            ]
            if tavily:
                lines.append(
                    f"Tavily (web search + JS-rendering extract): "
                    f"**{'live' if tavily.get('live') else 'mock'}** — "
                    f"{tavily.get('free_tier', '')}"
                )
                if not tavily.get("live"):
                    lines.append(
                        f"To go live, set `{tavily.get('key_env')}` and "
                        f"`{tavily.get('enable_env')}=1`."
                    )
            return "\n".join(lines)

        # Drift reports. Keyed on the distinctive shape, and placed before the generic markdown
        # branch - a branch keyed on a common field swallows a more specific one.
        if "mean_psi" in data and "overall" in data:
            top = data.get("top") or []
            worst = ", ".join(f"{t['feature']} {t['psi']}" for t in top[:3])
            lines = [
                f"Drift check: {data.get('overall', 'unknown')}.",
                f"Mean PSI across {data.get('n_features_scored', 0)} features is "
                f"{data.get('mean_psi', 0)} "
                f"({data.get('n_drifted', 0)} above the significant threshold, "
                f"{data.get('n_watching', 0)} in the watch band).",
            ]
            if worst:
                lines.append(f"Largest shifts: {worst}.")
            if data.get("features_missing_from_batch"):
                lines.append(
                    "Features absent from the batch: "
                    + ", ".join(data["features_missing_from_batch"][:5])
                    + " - a schema problem, not drift."
                )
            lines.append(
                "PSI is computed on binned marginals with bin edges fixed at training time; "
                "drift means the model is outside the data it was trained on, not that it is "
                "necessarily wrong."
            )
            return "\n".join(lines)

        if "markdown" in data and "backend" in data:
            if data.get("backend") == "error":
                return f"Rendering failed: {data.get('error', 'unknown error')}."
            body = " ".join(str(data.get("markdown", "")).split())[:400]
            banner = (
                ""
                if data.get("live")
                else "⚠ **Mock render — Firecrawl is not configured; no page was fetched.**\n\n"
            )
            return (
                f"{banner}Rendered `{data.get('url', '')}` via **{data.get('backend')}** "
                f"(HTTP {data.get('status', '?')}):\n\n{body}…"
            )

        if "markdown" in data or "content" in data:
            body = (data.get("markdown") or data.get("content") or "")[:300]
            return f"Scraped {data.get('url', 'the page')}:\n\n{body}…"

    return f"Tool returned:\n\n```json\n{tool_output[:1200]}\n```"


# --- factory -----------------------------------------------------------------


def resolve_provider_name(requested: str | None = None) -> str:
    """Which provider ``auto`` (or an explicit name) resolves to, without constructing it.

    Split out from :func:`get_provider` so the hub, the badge and the tests all answer
    "which model is actually answering?" from one place. Two implementations of this rule is
    exactly how the old deploy ended up advertising Groq while the chat ran on the mock.
    """
    name = (requested or os.getenv("SAROS_LLM", "auto")).strip().lower()
    if name == "auto":
        if os.getenv("GROQ_API_KEY", "").strip():
            return "groq"
        if os.getenv("GEMINI_API_KEY", "").strip():
            return "gemini"
        return "mock"
    if name in ("groq", "gemini", "mock", "echo"):
        return name
    # An unknown value is a typo, not a request for silence: fall back to auto's rule.
    return resolve_provider_name("auto")


def get_provider(name: str | None = None) -> Provider:
    """Resolve the provider from env. Never raises for a missing key — falls back to mock.

    Default is ``auto``: ``GROQ_API_KEY`` -> Groq, else ``GEMINI_API_KEY`` -> Gemini, else the
    labelled mock. A student deploying on a free tier will frequently have no key configured,
    and the hub should still be a working product rather than a 500 — but a key that *is*
    configured must never be ignored, which is what a hardcoded "mock" default did.

    An explicitly named provider that cannot be constructed (``SAROS_LLM=groq`` with no key)
    degrades to the mock rather than raising, so a bad env var cannot take the hub down.
    """
    name = resolve_provider_name(name)
    if name == "groq":
        try:
            return GroqProvider()
        except RuntimeError:
            return MockProvider()
    if name == "gemini":
        try:
            return GeminiProvider()
        except RuntimeError:
            return MockProvider()
    if name == "echo":
        return EchoProvider()
    return MockProvider()
