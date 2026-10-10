"""Knowledge RAG — a Nexus body.

Retrieval lives here; *synthesis* lives in the hub agent. That split is deliberate: a body
should not own an LLM dependency, and keeping retrieval separate means the same passages can
be cited by any provider.

What this body guarantees is provenance — every result carries its source file, heading and
relevance score, so an answer can always be traced back to the text it came from.
"""

from __future__ import annotations

import threading
from typing import Any

from core.contract import Module
from core.types import AgentTool, Explanation, MetricsReport, ModuleManifest, Prediction

from .retriever import Retriever


def corpus_status(r: "Retriever") -> dict[str, Any]:
    """Provenance of the loaded corpus, including any external-source failure.

    Surfaced rather than logged: a corpus silently missing documents is worse than a
    visible warning, because the answers still look confident.
    """
    from core.integrations.supabase import status as supabase_status

    st = supabase_status()
    out = {
        "external_backend": st["backend"],
        "external_live": st["live"],
        "n_external_passages": getattr(r, "_n_external", 0),
    }
    err = getattr(r, "_external_error", None)
    if err:
        out["external_error"] = err
        out["warning"] = (
            f"External corpus failed to load ({err}); answering from the in-repo "
            "documents only."
        )
    return out

ACCENT = "#38bdf8"

#: Words too common to count as evidence of a topical match.
_STOPWORDS = frozenset("""
a an and are as at be but by for from has have how i if in is it its of on or that the this to
was were what when where which who why with you your not can do does what's
""".split())


def _matched_term_count(query: str, text: str) -> int:
    """How many substantive query terms actually appear in the passage.

    TF-IDF cosine can be driven entirely by one shared token, which produces confident-looking
    matches on unrelated documents. Counting substantive overlaps is a cheap guard.
    """
    hay = text.lower()
    terms = {w for w in query.lower().split() if w.strip("?.!,;:") and w not in _STOPWORDS}
    return sum(1 for t in terms if t in hay)


def _grounding(query: str, hits: list[Any]) -> tuple[str, str, int]:
    """Return ``(label, decision, matched_terms)`` for a set of retrieval hits.

    One implementation, called by both ``predict`` and the ``search_knowledge`` tool, so the
    agent and the storefront can never disagree about whether an answer was grounded. This
    used to live only inside ``predict``, which meant the tool handed the agent raw scores
    with no verdict attached and it summarised them as a JSON dump.

    The discriminator is the count of substantive query terms found in the top passage, not
    the cosine. An earlier version floored the cosine at 0.25, but on a 25-passage corpus
    TF-IDF tops out near 0.22 even for obviously-relevant questions — the floor rejected
    every real query and labelled the whole body useless. Measured:

        "What discount can an agent approve?"        0.183   2 terms   relevant
        "What is the refund window?"                 0.175   1 term    relevant
        "airspeed velocity of an unladen swallow"    0.175   1 term    irrelevant

    Cosine cannot separate those last two. Term count can, so a single shared token is called
    weak even though it occasionally demotes a real question that the corpus answers at rank
    two. Being conservative about grounding is the cheaper error here.
    """
    top = hits[0].score if hits else 0.0
    matched_terms = _matched_term_count(query, hits[0].heading + " " + hits[0].text) if hits else 0

    if not hits or matched_terms == 0:
        label, decision = "No grounding", (
            "The corpus has nothing on this. Say so rather than answering from memory."
        )
    elif matched_terms < 2:
        label, decision = "Weak grounding", (
            "The match rests on a single shared word — treat any answer as unverified and "
            "ask a clarifying question."
        )
    else:
        label, decision = "Well grounded", (
            f"Answer from `{hits[0].source}.md` → {hits[0].heading} and cite it."
        )
    return label, decision, matched_terms



class RagModule(Module):
    id = "rag"

    def __init__(self) -> None:
        self._retriever: Retriever | None = None
        self._lock = threading.Lock()

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = Retriever().build()
        return self._retriever

    # --- contract ------------------------------------------------------------

    @property
    def manifest(self) -> ModuleManifest:
        r = self.retriever
        return ModuleManifest(
            id=self.id,
            title="Knowledge RAG",
            summary=(
                "Retrieval-augmented answers over an enterprise policy corpus. Heading-aware "
                "chunking, TF-IDF ranking, and mandatory citations so nothing is asserted "
                "without a source."
            ),
            accent=ACCENT,
            status="live",
            capabilities=["retrieval", "citations", "policy-lookup", "grounded-answers"],
            routes={"ui": "/rag", "api": "/api/v1/rag"},
            tools=self.tools(),
        )

    def predict(self, payload: dict[str, Any]) -> Prediction:
        """For RAG, `score` is retrieval confidence rather than a risk probability."""
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("Missing required features: query")

        k = int(payload.get("k", 5))
        hits = self.retriever.search(query, k=k)
        top = hits[0].score if hits else 0.0

        # A single shared word is not grounding. "What is the airspeed velocity of an unladen
        # swallow" matches the fraud protocol on the word "velocity" alone and scored 0.175,
        # which is why the floor sits at 0.25 and the matched-term count is reported.
        label, decision, matched_terms = _grounding(query, hits)

        return Prediction(
            module_id=self.id,
            score=round(min(1.0, top), 4),
            label=label,
            decision=decision,
            inputs={"k": float(k)},
            meta={
                "matched_terms": matched_terms,
                "n_passages": self.retriever.size,
                "sources": self.retriever.sources,
                "retriever": "tfidf(1,2)-sublinear",
                "hits": [
                    {"source": h.source, "heading": h.heading, "score": h.score, "snippet": h.snippet}
                    for h in hits
                ],
            },
        )

    def explain(self, payload: dict[str, Any]) -> Explanation:
        """Retrieval explanations are provenance, not attributions."""
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("Missing required features: query")
        hits = self.retriever.search(query, k=int(payload.get("k", 5)))
        return Explanation(
            kind="retrieval",
            space="unspecified",
            attributions=[],
            summary=(
                f"{len(hits)} passages matched. Strongest: `{hits[0].source}.md` → "
                f"{hits[0].heading}." if hits else "No passages matched the query."
            ),
            caveats=[
                "Retrieval is lexical TF-IDF, not semantic embeddings — synonyms and paraphrase "
                "will not match. A dense backend is a drop-in replacement behind the same interface.",
                "A high relevance score means lexical overlap, not that the passage answers the question.",
            ],
        )

    def evaluate(self) -> MetricsReport:
        """Retrieval quality on a small probe set, so the body is not unverifiable."""
        probes = [
            ("What is the refund window?", "refund-policy", 30),
            ("How much discount can an agent approve?", "retention-playbook", 20),
            ("What uptime do we commit to?", "sla", 99),
            ("Why are charge columns dropped?", "churn-model-card", 3),
            ("How long is usage telemetry kept?", "data-privacy", 24),
            ("What happens to a held transaction?", "fraud-protocol", 15),
        ]
        hit_at_1, hit_at_3, notes = 0, 0, []
        for query, expected_src, expected_needle in probes:
            hits = self.retriever.search(query, k=3)
            sources = [h.source for h in hits]
            hit_at_1 += int(bool(sources) and sources[0] == expected_src)
            hit_at_3 += int(expected_src in sources)
            blob = " ".join(h.text for h in hits).lower()
            if str(expected_needle) in blob:
                notes.append(f"'{query}' → correct passage containing '{expected_needle}'.")
            else:
                notes.append(f"'{query}' → retrieved {sources} but the key value was not present.")

        n = len(probes)
        return MetricsReport(
            n_samples=n,
            positive_rate=1.0,
            scores={
                "hit@1": hit_at_1 / n,
                "hit@3": hit_at_3 / n,
                "corpus_passages": float(self.retriever.size),
                "corpus_documents": float(len(self.retriever.sources)),
            },
            notes=[
                "Probe set is hand-written from the corpus, so hit@1 is a smoke test for the "
                "index, not an unbiased quality estimate.",
                *notes,
            ],
        )

    # --- agent tools ---------------------------------------------------------

    def tools(self) -> list[AgentTool]:
        return [
            AgentTool(
                name="search_knowledge",
                description=(
                    "Search the enterprise knowledge base (refund policy, retention playbook, "
                    "SLA, model cards, data privacy, fraud protocol). Returns ranked passages "
                    "with source files. Use this before answering any policy question."
                ),
                module_id=self.id,
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for."},
                        "k": {"type": "integer", "description": "Passages to return.", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            AgentTool(
                name="list_knowledge_sources",
                description="List the documents currently indexed in the knowledge base.",
                module_id=self.id,
                parameters_schema={"type": "object", "properties": {}},
            ),
        ]

    def tool_search_knowledge(self, query: str, k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, k=k)
        label, decision, matched_terms = _grounding(query, hits)
        return {
            "query": query,
            "n_results": len(hits),
            # The verdict travels with the passages. Without it the agent receives bare cosine
            # scores and has no way to tell a grounded answer from a coincidental one.
            "grounding": label,
            "matched_terms": matched_terms,
            "decision": decision,
            "results": [
                {"source": h.source, "heading": h.heading, "score": h.score, "snippet": h.snippet}
                for h in hits
            ],
            "citation": self.retriever.cite(hits),
        }

    def tool_list_knowledge_sources(self) -> dict[str, Any]:
        r = self.retriever
        out: dict[str, Any] = {
            "n_documents": len(r.sources),
            "n_passages": r.size,
            "sources": r.sources,
            # Where the corpus came from. Without this a Supabase outage looks identical to a
            # knowledge base that genuinely has six documents.
            "corpus": corpus_status(r),
        }
        return out


MODULE = RagModule()
