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

    def build(self) -> "Retriever":
        docs = sorted(self.corpus_dir.glob("*.md"))
        if not docs:
            raise FileNotFoundError(f"No markdown found in {self.corpus_dir}")

        for doc in docs:
            for head, text in _chunk(doc.read_text(encoding="utf-8")):
                self._passages.append((doc.stem, head, text))

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
        return self

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
            lines.append(f"- `{p.source}.md` → {p.heading} (relevance {p.score:.2f})")
        return "\n".join(lines)
