"""RAG retriever — TF-IDF over a chunked markdown corpus.

Deliberately lexical rather than embedded. Dense retrieval needs an embedding model, and on a
512 MB free-tier container that is the wrong trade: TF-IDF is exact, instant, needs no GPU and
no API key, and for a bounded policy corpus it retrieves the right passages.

The interface is the part that matters — :class:`Retriever` returns ranked passages with
provenance, so swapping in a vector backend later is one class, not a rewrite of the body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


@dataclass(frozen=True)
class Passage:
    source: str
    heading: str
    text: str
    score: float

    @property
    def snippet(self) -> str:
        flat = " ".join(self.text.split())
        return flat[:220] + ("…" if len(flat) > 220 else "")


def _chunk(markdown: str, max_chars: int = 900) -> list[tuple[str, str]]:
    """Split on headings, then merge small sections and split oversized ones.

    Chunking on structure rather than a fixed character window keeps a rule and its table in
    the same passage — which is the difference between a useful retrieval and a fragment.
    """
    sections: list[tuple[str, list[str]]] = []
    current_head, buf = "Overview", []
    for line in markdown.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((current_head, buf))
            current_head, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((current_head, buf))

    chunks: list[tuple[str, str]] = []
    for head, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append((head, text))
        else:
            # Split on blank lines so we break between paragraphs, not mid-sentence.
            parts, cur = [], ""
            for para in re.split(r"\n\s*\n", text):
                if cur and len(cur) + len(para) > max_chars:
                    parts.append(cur.strip())
                    cur = para
                else:
                    cur = f"{cur}\n\n{para}" if cur else para
            if cur.strip():
                parts.append(cur.strip())
            for i, part in enumerate(parts, 1):
                chunks.append((f"{head} (part {i})", part))
    return chunks


class Retriever:
    """TF-IDF index over the corpus. Built once, reused."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self._passages: list[tuple[str, str, str]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        #: Passages that came from the external source rather than the in-repo markdown.
        self._n_external = 0
        #: Set when the external source was configured but failed. Surfaced rather than
        #: swallowed: a corpus silently missing documents is worse than a visible warning.
        self._external_error: str | None = None

    def build(self) -> "Retriever":
        docs = sorted(self.corpus_dir.glob("*.md"))
        if not docs:
            raise FileNotFoundError(f"No markdown found in {self.corpus_dir}")

        for doc in docs:
            for head, text in _chunk(doc.read_text(encoding="utf-8")):
                self._passages.append((doc.stem, head, text))

        n_local = len(self._passages)

        # Optional second source: rows in Supabase, merged with the in-repo documents. The
        # in-repo corpus is already loaded at this point, so a failure here degrades to "the
        # documents I shipped with" rather than to an empty knowledge base. That asymmetry is
        # the reason this is additive and wrapped, not a replacement for the files.
        try:
            from core.integrations.supabase import get_source

            extra = get_source().fetch_documents()
        except Exception as exc:  # noqa: BLE001 - a DB outage must not empty the corpus
            self._external_error = f"{type(exc).__name__}: {exc}"
            extra = []
        for row in extra:
            for head, text in _chunk(f"## {row['heading']}\n\n{row['body']}" if row.get("heading") else row["body"]):
                self._passages.append((row["source"], head, text))
        self._n_external = len(self._passages) - n_local

        self._index()
        return self

    def _index(self) -> None:
        """Build the lexical index for the current passage list."""
        # sublinear_tf damps the advantage of long passages; ngram_range picks up phrases like
        # "customer service" that unigrams alone would split.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform([t for _, _, t in self._passages])

    def with_documents(self, documents: list[tuple[str, str]]) -> "Retriever":
        """Return a separately indexed copy augmented with these session documents.

        The base retriever is never mutated, so an upload from one browser session cannot leak
        into another session's search results.
        """
        overlay = Retriever(self.corpus_dir)
        overlay._passages = list(self._passages)
        overlay._n_external = self._n_external
        overlay._external_error = self._external_error
        for source, content in documents:
            for heading, text in _chunk(content):
                overlay._passages.append((source, heading, text))
        overlay._index()
        return overlay

    @property
    def size(self) -> int:
        return len(self._passages)

    @property
    def sources(self) -> list[str]:
        return sorted({src for src, _, _ in self._passages})

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if self._matrix is None or self._vectorizer is None:
            raise RuntimeError("Retriever has not been built.")
        if not query.strip():
            return []

        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = sims.argsort()[::-1]

        out: list[Passage] = []
        for idx in order[:k]:
            score = float(sims[idx])
            if score <= 0.0:
                break
            src, head, text = self._passages[int(idx)]
            out.append(Passage(source=src, heading=head, text=text, score=round(score, 4)))
        return out

    def cite(self, passages: list[Passage]) -> str:
        """A provenance block. Retrieval without citations is how RAG systems lie confidently."""
        if not passages:
            return "_No passages matched; this answer is not grounded in the knowledge base._"
        lines = ["**Sources**"]
        for p in passages:
            source = p.source if Path(p.source).suffix else f"{p.source}.md"
            lines.append(f"- `{source}` → {p.heading} (relevance {p.score:.2f})")
        return "\n".join(lines)
