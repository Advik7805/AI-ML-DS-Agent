"""Knowledge RAG — a Saros body.

Retrieval lives here; *synthesis* lives in the hub agent. That split is deliberate: a body
should not own an LLM dependency, and keeping retrieval separate means the same passages can
be cited by any provider.

What this body guarantees is provenance — every result carries its source file, heading and
relevance score, so an answer can always be traced back to the text it came from.
"""

from __future__ import annotations

import threading
from io import BytesIO
from pathlib import Path
from typing import Any

from core.contract import Module
from core.types import AgentTool, Explanation, MetricsReport, ModuleManifest, Prediction, UploadAnalysis
from core.upload_utils import MAX_RETURNED_PREDICTIONS

from .retriever import Retriever
from .session_corpus import SESSION_CORPUS


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
MAX_DOCUMENT_TEXT_BYTES = 2_000_000

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


def _source_reference(source: str) -> str:
    """Keep uploaded filenames intact; built-in documents are source stems ending in .md."""
    return source if Path(source).suffix else f"{source}.md"


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
            f"Answer from `{_source_reference(hits[0].source)}` → {hits[0].heading} and cite it."
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
        return self.predict_with_retriever(query, self.retriever, int(payload.get("k", 5)))

    def predict_with_retriever(self, query: str, retriever: Retriever, k: int = 5) -> Prediction:
        """Score a query against a base or session-augmented retriever."""
        query = query.strip()
        if not query:
            raise ValueError("Missing required features: query")
        hits = retriever.search(query, k=k)
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
                "n_passages": retriever.size,
                "sources": retriever.sources,
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
                f"{len(hits)} passages matched. Strongest: `{_source_reference(hits[0].source)}` → "
                f"{hits[0].heading}." if hits else "No passages matched the query."
            ),
            caveats=[
                "Retrieval is lexical TF-IDF, not semantic embeddings — synonyms and paraphrase "
                "will not match. A dense backend is a drop-in replacement behind the same interface.",
                "A high relevance score means lexical overlap, not that the passage answers the question.",
            ],
        )

    def analyze_upload(
        self,
        filename: str | None,
        content: bytes | None,
        *,
        session_id: str | None = None,
        query: str = "",
    ) -> UploadAnalysis:
        stored_filename: str | None = filename
        active_session = session_id
        if content is not None:
            if not filename:
                raise ValueError("RAG uploads need a filename with a .md, .txt, or .pdf extension.")
            suffix = Path(filename).suffix.lower()
            if suffix in {".md", ".txt"}:
                try:
                    text = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    raise ValueError("RAG markdown/text uploads must be UTF-8 encoded.") from None
            elif suffix == ".pdf":
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(BytesIO(content), strict=False)
                    pages: list[str] = []
                    text_bytes = 0
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text_bytes += len(page_text.encode("utf-8"))
                        if text_bytes > MAX_DOCUMENT_TEXT_BYTES:
                            raise ValueError("Extracted document text exceeds the 2 MB per-document limit.")
                        pages.append(page_text)
                    text = "\n\n".join(pages)
                except ValueError:
                    raise
                except Exception as exc:  # noqa: BLE001 - malformed PDFs are user input
                    raise ValueError(f"Could not extract text from PDF: {type(exc).__name__}: {exc}") from None
            else:
                raise ValueError("RAG accepts .md, .txt, or .pdf uploads.")
            if len(text.encode("utf-8")) > MAX_DOCUMENT_TEXT_BYTES:
                raise ValueError("Extracted document text exceeds the 2 MB per-document limit.")
            active_session, stored_filename = SESSION_CORPUS.append(active_session, filename, text)

        documents = SESSION_CORPUS.documents(active_session)
        if not documents:
            raise ValueError("Upload a .md, .txt, or .pdf file before searching this session.")

        overlay = self.retriever.with_documents(documents)
        predictions: list[Prediction] = []
        notes = [
            f"The session index contains {len(documents)} uploaded document(s) and {overlay.size} passages, including the built-in knowledge base.",
            "Documents are held in bounded process memory only; they are not persisted and are isolated by session id.",
        ]
        scores = {
            "uploaded_documents": float(len(documents)),
            "indexed_passages": float(overlay.size),
        }
        positive_rate = 0.0
        if query.strip():
            prediction = self.predict_with_retriever(query, overlay)
            predictions.append(prediction)
            scores["query_match_score"] = float(prediction.score)
            scores["matched_terms"] = float(prediction.meta.get("matched_terms", 0))
            positive_rate = float(prediction.label == "Well grounded")
            notes.append(f"Search status: {prediction.label}. {prediction.decision}")
            notes.append("For this single query, positive_rate is 1 only when the grounding verdict is well grounded.")

        return UploadAnalysis(
            module_id=self.id,
            filename=stored_filename,
            n_rows=len(documents),
            metrics=MetricsReport(
                n_samples=len(documents),
                positive_rate=positive_rate,
                scores=scores,
                notes=notes,
            ),
            predictions=predictions[:MAX_RETURNED_PREDICTIONS],
            session_id=active_session,
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
