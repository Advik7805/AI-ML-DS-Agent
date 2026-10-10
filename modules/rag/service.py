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
        matched_terms = _matched_term_count(query, hits[0].text) if hits else 0
        if not hits:
            label, decision = "No grounding", (
                "The corpus has nothing on this. Say so rather than answering from memory."
            )
        elif top < 0.25 or matched_terms < 2:
            label, decision = "Weak grounding", (
                "Best match is weak or rests on a single shared word — treat any answer as "
                "unverified and ask a clarifying question."
            )
        else:
            label, decision = "Well grounded", (
                f"Answer from `{hits[0].source}.md` → {hits[0].heading} and cite it."
            )

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
        return {
            "query": query,
            "n_results": len(hits),
            "results": [
                {"source": h.source, "heading": h.heading, "score": h.score, "snippet": h.snippet}
                for h in hits
            ],
            "citation": self.retriever.cite(hits),
        }

    def tool_list_knowledge_sources(self) -> dict[str, Any]:
        r = self.retriever
        return {"n_documents": len(r.sources), "n_passages": r.size, "sources": r.sources}


MODULE = RagModule()
