"""Analyze-your-own-data: parsing, validating and scoring an uploaded file.

Three properties this module is responsible for, because getting any of them wrong turns a
demo into a lie:

1. **Never persisted.** Everything happens on an in-memory buffer and the result is returned.
   The file is not written to ``data/``, not appended to the shipped corpus, and not cached
   between sessions. See :mod:`core.sessions`.
2. **Capped.** 10 MB per file, rows bounded before scoring, predictions returned for the top N.
   Free-tier instances have 512 MB and one worker; an unbounded parse is a scheduled outage.
3. **Specific when it refuses.** A wrong-columns CSV does not get "invalid file". It gets the
   columns the body needs, the columns it found, and which of the found ones were ignored. That
   message is the entire difference between a usable tool and a frustrating one.

Scoring is vectorised over the whole file (fast) and then the top N rows are re-scored through
the body's normal ``predict`` so the returned records are the *same shape* the existing UI
components already render — with SHAP drivers and a decision, not a bare probability.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from core.sessions import MAX_FILE_BYTES, Session
from core.types import ColumnsReport, DatasetCard, MetricsReport, Prediction, UploadAnalysis

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import at runtime
    from core.contract import Module

#: How many rows are scored, regardless of file size. Beyond this the marginal information is
#: negligible and the cost is not.
MAX_ROWS = 20_000
#: How many rows come back as full Prediction objects (with SHAP factors).
MAX_PREDICTIONS = 25
#: Rows echoed back for the preview table.
MAX_PREVIEW = 5


class UploadRejected(ValueError):
    """A refusal that tells the user exactly what to change.

    ``detail`` is what the API returns: a message plus the expected/found column lists, so the
    frontend can show a mapping table rather than a paragraph.
    """

    def __init__(
        self,
        message: str,
        *,
        expected: list[str] | None = None,
        found: list[str] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.expected = expected or []
        self.found = found or []
        self.hint = hint or ""

    def as_detail(self) -> dict[str, Any]:
        out: dict[str, Any] = {"message": self.message}
        if self.expected:
            out["expected_columns"] = list(self.expected)
        if self.found:
            out["found_columns"] = list(self.found)
        if self.hint:
            out["hint"] = self.hint
        return out


# --- parsing --------------------------------------------------------------------


def kind_of(filename: str, data: bytes) -> str:
    """Classify by extension, then verify by content. Extensions lie constantly."""
    name = (filename or "").lower()
    if data[:5] == b"%PDF-":
        return "pdf"
    if name.endswith(".csv") or name.endswith(".tsv"):
        return "csv"
    if name.endswith((".md", ".markdown")):
        return "markdown"
    if name.endswith((".txt", ".log", ".dat")):
        # Turbofan telemetry arrives as space-aligned .txt, so let the caller decide by shape —
        # this is only a first guess.
        return "text"
    # Fall back to guessing from the first line.
    head = data[:4096].decode("utf-8", errors="ignore")
    if "," in head.splitlines()[0] if head.splitlines() else False:
        return "csv"
    return "unknown"


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip('"').strip("\ufeff") for c in df.columns]
    drop = [c for c in df.columns if not c or c.lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)
    return df


def read_csv_bytes(data: bytes, *, separator: str | None = None) -> pd.DataFrame:
    """Parse CSV/TSV from memory, with a readable error instead of a pandas traceback."""
    # Explicit separators only, tried in order of likelihood, and the sniffing engine is never
    # allowed to win on column count alone. A two-line file with a single "url" column was sniffed
    # as u-delimited and returned columns ['Unnamed: 0', 'rl'] — a *worse* parse that happened to
    # have more columns. Automatic detection is a helpful guess, not an authority.
    attempts: list[str | None] = [separator] if separator else [",", "\t", ";"]
    parsed: pd.DataFrame | None = None
    for sep in attempts:
        try:
            if sep is None:
                candidate = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
            else:
                candidate = pd.read_csv(io.BytesIO(data), sep=sep)
        except UnicodeDecodeError as exc:
            raise UploadRejected(
                "The file is not valid UTF-8 text. Export it as UTF-8 CSV and try again.",
                hint="Excel: 'Save as' → 'CSV UTF-8'. A .xlsx workbook must be exported to CSV first.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - pandas raises several unrelated types here
            if parsed is None and sep == attempts[0]:
                raise UploadRejected(f"Could not parse the file as CSV: {type(exc).__name__}: {exc}") from exc
            continue
        if parsed is None or candidate.shape[1] > parsed.shape[1]:
            parsed = candidate
        if candidate.shape[1] >= 2:
            break

    # Last resort for a space-aligned file: a header with several whitespace-separated tokens but
    # only one parsed column means the delimiter was whitespace, not comma.
    if parsed is not None and parsed.shape[1] == 1:
        first_line = data[:4096].splitlines()[0].decode("utf-8", errors="ignore") if data.strip() else ""
        if len(first_line.split()) > 1:
            try:
                candidate = pd.read_csv(io.BytesIO(data), sep=r"\s+", engine="python")
                if candidate.shape[1] > parsed.shape[1]:
                    parsed = candidate
            except Exception:  # noqa: BLE001 - the single-column parse already succeeded
                pass

    if parsed is None or parsed.empty:
        raise UploadRejected("The file parsed but contains no rows.")
    return _coerce_columns(parsed)


def read_whitespace_table(data: bytes, names: list[str]) -> pd.DataFrame:
    """NASA CMAPSS style: space-aligned, no header, 26 columns."""
    try:
        df = pd.read_csv(io.BytesIO(data), sep=r"\s+", header=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise UploadRejected(
            f"Could not parse the file as whitespace-delimited telemetry: {exc}"
        ) from exc
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(names):
        from modules.maintenance.data import cmapss_columns

        # The real FD001 file has 26 columns (3 settings); the generator emits 29 (6 settings).
        # Accepting only one of those made "drop in the real file" a promise the code broke.
        alt = cmapss_columns(df.shape[1])
        if alt is None:
            raise UploadRejected(
                f"Expected {len(names)} whitespace-separated columns (unit, cycle, settings, "
                f"21 sensors) but found {df.shape[1]}.",
                expected=names,
                found=[f"column {i + 1}" for i in range(df.shape[1])],
                hint=(
                    "This is the NASA CMAPSS layout: no header row, unit and cycle, the "
                    "operational settings, then 21 sensor readings. The real FD001 file has 26 "
                    "columns and is accepted; a CSV whose header names the columns is accepted too."
                ),
            )
        names = alt
    df.columns = names
    return df


def extract_text(filename: str, data: bytes) -> str:
    """Plain text from .md/.txt/.pdf. PDFs need pypdf; when it is missing, say so explicitly."""
    if data[:5] == b"%PDF-":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on the deploy image
            raise UploadRejected(
                "PDF support requires the pypdf package, which is not installed in this image.",
                hint="Install pypdf (it is in requirements.txt), or upload the document as .md/.txt.",
            ) from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages[:60]]
        except Exception as exc:  # noqa: BLE001 - a corrupt PDF must not 500
            raise UploadRejected(f"Could not read the PDF: {type(exc).__name__}: {exc}") from exc
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text.strip():
            raise UploadRejected(
                "The PDF has no extractable text layer — it is probably a scan.",
                hint="Scanned documents need OCR; upload a text export of the document instead.",
            )
        return text

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# --- generic pandas helpers ------------------------------------------------------


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case/space-insensitive column lookup, so ``Customer ID`` finds ``customer_id``."""
    normalised = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        hit = normalised.get(re.sub(r"[^a-z0-9]", "", cand.lower()))
        if hit:
            return hit
    return None


def validate_columns(
    df: pd.DataFrame,
    required: list[str],
    optional: list[str],
    *,
    also_accept: dict[str, list[str]] | None = None,
) -> ColumnsReport:
    """Check the file against the body's schema and explain any gap.

    ``also_accept`` maps a required column to synonyms that satisfy it (e.g. the churn target
    ``churn`` is also spelled ``Churn``, ``churned``, ``is_churn``). Synonyms are how a real
    analyst's export actually looks.
    """
    also_accept = also_accept or {}
    found = [str(c) for c in df.columns]
    matched: list[str] = []
    missing: list[str] = []
    for col in required:
        if find_column(df, [col, *also_accept.get(col, [])]) is not None:
            matched.append(col)
        else:
            missing.append(col)
    if missing:
        raise UploadRejected(
            f"This body needs {len(required)} columns and the file is missing {len(missing)}: "
            f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}.",
            expected=required,
            found=found,
            hint=(
                "Column names are matched case- and punctuation-insensitively, so "
                "'Total Day Minutes' and 'total_day_minutes' are the same column. Everything "
                "else in the file is ignored rather than rejected."
            ),
        )
    extra = [c for c in found if c not in matched and c not in optional]
    return ColumnsReport(
        expected=required,
        matched=matched,
        missing=[],
        ignored=extra[:40],
        recognised_optional=[c for c in optional if find_column(df, [c])],
    )


# --- metrics over an uploaded file ----------------------------------------------


def _distribution_report(values: np.ndarray, notes: list[str]) -> MetricsReport:
    """Unlabelled upload: report the score distribution and say plainly there is no accuracy.

    Inventing a "quality" number here would be the single most misleading thing this endpoint
    could do — there is no ground truth in an unlabelled file, so any accuracy figure would be
    computed against the model's own output.
    """
    v = np.asarray(values, dtype=float)
    deciles = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]) if v.size else np.zeros(5)
    return MetricsReport(
        n_samples=int(v.size),
        positive_rate=float(v.mean()) if v.size else 0.0,
        scores={
            "mean_score": float(v.mean()) if v.size else 0.0,
            "p10": float(deciles[0]),
            "median": float(deciles[2]),
            "p90": float(deciles[4]),
            "max_score": float(v.max()) if v.size else 0.0,
        },
        notes=[
            *notes,
            "No label column was found, so no accuracy metric is reported. Every quality number "
            "on this panel would otherwise be the model grading its own homework.",
            "The distribution above is what the model produced on your file; it is not evidence "
            "that the model is right about your file.",
        ],
    )


def _labelled_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    notes: list[str],
) -> MetricsReport:
    """Real held-out metrics when the upload carries its own labels."""
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
    baseline = float(y_true.mean()) if y_true.size else 0.0
    scores_out: dict[str, float] = {"baseline_pr_auc": round(baseline, 4)}
    if len(set(y_true.tolist())) > 1:
        scores_out["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
        scores_out["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
    else:
        notes = [
            *notes,
            "The label column has only one class in your file, so PR-AUC and ROC-AUC are "
            "undefined and deliberately not reported.",
        ]
    scores_out["precision"] = round(float(precision_score(y_true, preds, zero_division=0)), 4)
    scores_out["recall"] = round(float(recall_score(y_true, preds, zero_division=0)), 4)
    if y_true.size < 200:
        notes = [
            *notes,
            f"Only {y_true.size} labelled rows: these metrics have wide confidence intervals and "
            "should not be read as the model's quality on your traffic. Upload a fuller extract "
            "before drawing a conclusion.",
        ]
    if scores_out.get("pr_auc", 0.0) >= 0.999:
        notes = [
            *notes,
            "PR-AUC of 1.000 on an uploaded sample almost always means the file is too small or "
            "too easy, not that the model is perfect. Check the positive count in the confusion "
            "matrix before believing it.",
        ]
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return MetricsReport(
        n_samples=int(y_true.size),
        positive_rate=round(baseline, 4),
        scores=scores_out,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        threshold=float(threshold),
        notes=[
            *notes,
            "Metrics are computed on *your* labels at the model's shipped operating threshold. "
            "They describe how this model behaves on your file, which is the question worth "
            "asking before trusting it on your production traffic.",
        ],
    )


# --- result assembly ------------------------------------------------------------


def upload_dataset_card(module: "Module", filename: str, rows: int | None, columns: list[str]) -> DatasetCard:
    """A provenance card for the *upload*, so results never inherit the shipped dataset's card."""
    base = module.dataset_card()
    return base.model_copy(
        update={
            "dataset": f"Your upload — {filename}",
            "origin": "uploaded",
            "origin_detail": (
                f"Uploaded by this session and processed in memory: {rows if rows is not None else 'n/a'} rows, "
                f"{len(columns)} columns. Never written to disk and never merged into the shipped corpus. "
                "The model scoring it is still the body's own model, fitted on the tenant data below."
            ),
            "origin_url": "",
            "file": filename,
            "rows": rows,
            "observed_rows": rows,
            "observed_source": "uploaded",
            "observed_basis": "uploaded file",
            "system_of_record": "your file (in-memory only)",
            "connect": (
                f"Scored by the {base.business_unit} model, so the result is directly comparable "
                "with the shipped tenant data on the body's own pages."
            ),
            "caveats": [
                *base.caveats[:1],
                "Nothing about this file is stored: reload the page with a new session and it is gone.",
            ],
        }
    )


def finish(
    module: "Module",
    *,
    filename: str,
    size_bytes: int,
    kind: str,
    df: pd.DataFrame | None,
    scores: np.ndarray,
    threshold: float,
    label: np.ndarray | None,
    metrics: MetricsReport,
    extra_notes: list[str],
    session_id: str,
) -> UploadAnalysis:
    """Score the top rows through the body's real ``predict`` and assemble the response."""
    predictions: list[Prediction] = []
    if df is not None and scores.size:
        order = np.argsort(scores)[::-1][:MAX_PREDICTIONS]
        for idx in order:
            row = df.iloc[int(idx)]
            payload = {
                str(k): (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v))
                for k, v in row.items()
            }
            try:
                pred = module.predict(payload)
            except Exception:  # noqa: BLE001 - one unusable row must not sink the whole file
                continue
            pred.meta["upload_row"] = int(idx)
            predictions.append(pred)

    columns = [] if df is None else [str(c) for c in df.columns]
    preview: list[dict[str, Any]] = []
    if df is not None:
        head = df.head(MAX_PREVIEW)
        preview = [
            {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()} for _, row in head.iterrows()
        ]

    return UploadAnalysis(
        session_id=session_id,
        module_id=module.id,
        filename=filename,
        size_bytes=size_bytes,
        kind=kind,
        rows=None if df is None else int(len(df)),
        columns_report=ColumnsReport(
            expected=[], matched=list(columns), missing=[], ignored=[], recognised_optional=[]
        ),
        threshold=threshold,
        metrics=metrics,
        predictions=predictions,
        dataset=upload_dataset_card(module, filename, None if df is None else int(len(df)), columns),
        preview=preview,
        notes=extra_notes,
    )


# --- the generic CSV path -------------------------------------------------------


def _canonicalise(df: pd.DataFrame, required: list[str], optional: list[str], aliases: dict[str, list[str]]) -> pd.DataFrame:
    """Rename the file's columns to the body's names, matched case/punctuation-insensitively.

    A user who exports ``Total Day Minutes`` and a body that trains on ``total_day_minutes`` are
    describing the same data. Refusing one because of spacing would be pedantry that costs the
    user their upload.
    """
    rename: dict[str, str] = {}
    for col in [*required, *optional]:
        hit = find_column(df, [col, *aliases.get(col, [])])
        if hit is not None and hit != col:
            rename[hit] = col
    return df.rename(columns=rename)


def _binary(series: pd.Series) -> np.ndarray:
    """Labels arrive as 1/0, true/false, yes/no, churn/not. All four are the same fact."""
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {"1": 1, "1.0": 1, "true": 1, "yes": 1, "y": 1, "churn": 1, "fraud": 1, "positive": 1,
         "0": 0, "0.0": 0, "false": 0, "no": 0, "n": 0, "not churn": 0, "legit": 0, "negative": 0}
    )
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna((numeric > 0.5).astype(float)).fillna(0)
    return mapped.astype(int).to_numpy()


def analyze_generic(module: "Module", filename: str, data: bytes, session: Session, spec: dict[str, Any]) -> UploadAnalysis:
    """Validate, score and report on a tabular upload using the body's own model."""
    required = list(spec.get("required") or [])
    optional = list(spec.get("optional") or [])
    features = list(spec.get("features") or required)
    aliases = dict(spec.get("aliases") or {})
    notes = list(spec.get("notes") or [])

    if spec.get("kind") == "telemetry":
        # Headerless space-aligned telemetry (CMAPSS layout) unless the file has a header line.
        first = data[:2048].decode("utf-8", errors="ignore").splitlines()[0] if data.strip() else ""
        names = list(spec.get("names") or [])
        headerish = any(re.search(r"[A-Za-z]{3,}", tok) for tok in first.replace(",", " ").split())
        if headerish and ("," in first or "\t" in first):
            df = read_csv_bytes(data, separator="," if "," in first else "\t")
        elif headerish:
            df = read_whitespace_table(data, names) if names else read_csv_bytes(data)
        else:
            df = read_whitespace_table(data, names) if names else read_csv_bytes(data)
    else:
        df = read_csv_bytes(data, separator="\t" if filename.lower().endswith(".tsv") else None)

    # Some sources legitimately omit columns the model carries as constants: the real NASA FD001
    # file ships 3 of the 6 settings this repo's generator emits. Padding them with the trained
    # constant is correct — and it is reported in the notes, not done silently.
    padded = [c for c in spec.get("pad", []) if c not in df.columns]
    if padded:
        for col in padded:
            df[col] = float(spec.get("pad_value", 0.0))
        notes.append(
            f"{len(padded)} constant column(s) absent from your file were filled with the value "
            f"the model was trained on ({', '.join(padded)}). The real FD001 file ships three "
            "operational settings where this schema carries six."
        )

    report = validate_columns(df, required, optional, also_accept=aliases)
    df = _canonicalise(df, required, optional, aliases)
    # Features only. Coercing the label column to numeric turned "yes"/"no" into NaN, which made
    # a perfectly labelled file look unlabelled and dropped the user's ground truth on the floor.
    df = to_numeric(df, features)
    if len(df) > MAX_ROWS:
        notes.append(
            f"The file has {len(df):,} rows; the first {MAX_ROWS:,} were scored. Analysis is "
            "capped to keep a free-tier instance responsive."
        )
        df = df.head(MAX_ROWS)

    X = df[features]
    if X.isna().all(axis=1).any():
        notes.append(
            "Rows whose features could not be parsed as numbers were scored with NaN, which the "
            "model imputes. Check the preview for rows with blank cells."
        )
    scores, meta = module.score_frame(X)
    scores_arr = np.asarray(scores, dtype=float)
    threshold = float(getattr(module.bundle, "threshold", 0.5))
    notes.extend(meta.get("notes", []))

    label_col = spec.get("label")
    y = None
    if label_col and label_col in df.columns and df[label_col].notna().any():
        y = _binary(df[label_col])
        metrics = _labelled_report(y, scores_arr, threshold, notes)
    else:
        metrics = _distribution_report(scores_arr, notes)

    result = finish(
        module,
        filename=filename,
        size_bytes=len(data),
        kind=spec.get("kind", "csv"),
        df=df,
        scores=scores_arr,
        threshold=threshold,
        label=y,
        metrics=metrics,
        extra_notes=notes,
        session_id=session.id,
    )
    result.columns_report = report
    result.extra = {
        "accepts": spec.get("accepts", ""),
        "features_used": features,
        "scoring": "vectorised over the whole file; top rows re-scored with explanations",
        **meta,
    }
    return result


# --- RAG overlay ----------------------------------------------------------------


def chunk_document(text: str) -> list[tuple[str, str]]:
    """Heading-aware chunking, reusing the retriever's splitter so headings behave identically."""
    from modules.rag.retriever import _chunk  # local import: keeps core free of a hard dep

    chunks = _chunk(text)
    if not chunks:
        raise UploadRejected("The document has no extractable text after chunking.")
    return chunks


def overlay_search(session: Session, query: str, k: int = 4) -> list[dict[str, Any]]:
    """Retrieve from *this session's* uploaded documents, with citations to the user's filename.

    A separate index from the shipped corpus on purpose: scores from two TF-IDF spaces are not
    comparable, so they are reported as two labelled lists rather than merged into one ranking
    that would silently be nonsense.
    """
    passages = session.rag_passages
    if not passages or not query.strip():
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [t for _, _, t in passages]
    try:
        vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(texts)
        sims = cosine_similarity(vec.transform([query]), matrix)[0]
    except ValueError:
        # Empty vocabulary: the query shares no token with the overlay. Zero hits is the honest
        # answer, not a fallback to arbitrary passages.
        return []

    out: list[dict[str, Any]] = []
    for idx in sims.argsort()[::-1][:k]:
        score = float(sims[int(idx)])
        if score <= 0.0:
            break
        source, heading, text = passages[int(idx)]
        out.append(
            {
                "source": source,
                "heading": heading,
                "score": round(score, 4),
                "snippet": " ".join(text.split())[:280],
            }
        )
    return out


def describe_upload_limits() -> dict[str, Any]:
    return {
        "max_file_bytes": MAX_FILE_BYTES,
        "max_file_mb": round(MAX_FILE_BYTES / (1024 * 1024), 1),
        "max_rows_scored": MAX_ROWS,
        "max_predictions_returned": MAX_PREDICTIONS,
        "persisted": False,
    }


__all__ = [
    "UploadRejected",
    "MAX_FILE_BYTES",
    "MAX_ROWS",
    "MAX_PREDICTIONS",
    "analyze_generic",
    "chunk_document",
    "describe_upload_limits",
    "extract_text",
    "find_column",
    "finish",
    "kind_of",
    "overlay_search",
    "read_csv_bytes",
    "read_whitespace_table",
    "to_numeric",
    "upload_dataset_card",
    "validate_columns",
    "_distribution_report",
    "_labelled_report",
]
