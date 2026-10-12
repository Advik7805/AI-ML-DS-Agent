"use client";

import { useEffect, useRef, useState, type DragEvent, type ChangeEvent } from "react";

import { WakeRetry } from "@/components/body/WakeRetry";
import { clientFetch } from "@/lib/api";
import type { Prediction, UploadAnalysis } from "@/lib/types";

const MAX_FILE_BYTES = 10 * 1024 * 1024;

const ANALYZERS = [
  {
    id: "churn",
    title: "Churn & retention",
    accept: ".csv,text/csv",
    formats: "CSV · telecom customer features, optional churn label",
    accent: "#f59e0b",
  },
  {
    id: "fraud",
    title: "Fraud & anomaly",
    accept: ".csv,text/csv",
    formats: "CSV · transaction features, optional is_fraud label",
    accent: "#f43f5e",
  },
  {
    id: "pricing",
    title: "Dynamic pricing",
    accept: ".csv,text/csv",
    formats: "CSV · sku, price, demand (10+ rows per SKU)",
    accent: "#34d399",
  },
  {
    id: "maintenance",
    title: "Predictive maintenance",
    accept: ".csv,.txt,text/csv,text/plain",
    formats: "CSV or TXT · turbofan telemetry with unit, cycle, settings and sensors",
    accent: "#22d3ee",
  },
  {
    id: "rag",
    title: "Knowledge search",
    accept: ".md,.txt,.pdf,text/markdown,text/plain,application/pdf",
    formats: "Markdown, text or PDF · 2 MB extracted-text cap; session-only overlay",
    accent: "#38bdf8",
  },
] as const;

type AnalyzerId = (typeof ANALYZERS)[number]["id"];

function createSessionId() {
  return window.crypto?.randomUUID?.() ?? `saros-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getOrCreateSessionId() {
  try {
    const saved = window.localStorage.getItem("saros-rag-session");
    if (saved) return saved;
  } catch {
    // Keep the session usable for this mounted page even if browser storage is disabled.
  }
  return createSessionId();
}

function saveSessionId(sessionId: string) {
  try {
    window.localStorage.setItem("saros-rag-session", sessionId);
  } catch {
    // The in-memory React state still supports queries until this page is closed.
  }
}

function saveSessionSources(sources: string[]) {
  try {
    window.localStorage.setItem("saros-session-sources", JSON.stringify(sources));
  } catch {
    // This is only a convenience label; the server remains the source of truth.
  }
}

export function AnalyzeUpload() {
  const [moduleId, setModuleId] = useState<AnalyzerId>("churn");
  const [file, setFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [sourceLabel, setSourceLabel] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadAnalysis | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const analyzer = ANALYZERS.find((item) => item.id === moduleId)!;
  const ragCanQuery = moduleId === "rag" && Boolean(sessionId);

  useEffect(() => {
    const id = getOrCreateSessionId();
    setSessionId(id);
    saveSessionId(id);
  }, []);

  function choose(next: File | null) {
    setError(null);
    setResult(null);
    if (next && next.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("Choose a file smaller than 10 MB.");
      return;
    }
    setFile(next);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    choose(event.target.files?.[0] ?? null);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files?.[0] ?? null);
  }

  async function analyze() {
    if (!file && !ragCanQuery) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      const id = sessionId || getOrCreateSessionId();
      setSessionId(id);
      saveSessionId(id);
      form.append("session_id", id);
      if (sourceLabel.trim()) form.append("source_label", sourceLabel.trim());
      if (moduleId === "rag") form.append("query", query);
      const response = await clientFetch<UploadAnalysis>(`/api/v1/modules/${moduleId}/analyze`, {
        method: "POST",
        body: form,
      });
      setResult(response);
      if (response.session_id) {
        setSessionId(response.session_id);
        saveSessionId(response.session_id);
      }
      saveSessionSources(response.session_sources ?? []);
      window.dispatchEvent(new CustomEvent("saros:session-updated", {
        detail: { sessionId: response.session_id ?? id, sources: response.session_sources ?? [] },
      }));
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <div className="glass rounded-2xl p-5 sm:p-6">
        <label htmlFor="analyzer" className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
          Choose a body
        </label>
        <select
          id="analyzer"
          value={moduleId}
          onChange={(event) => {
            setModuleId(event.target.value as AnalyzerId);
            setFile(null);
            setError(null);
            setResult(null);
          }}
          className="mt-2 w-full rounded-xl border border-line bg-void-3 px-3.5 py-3 text-sm text-ink outline-none focus:border-hub/60"
        >
          {ANALYZERS.map((item) => (
            <option key={item.id} value={item.id}>
              {item.title}
            </option>
          ))}
        </select>

        <div className="mt-4 flex items-center justify-between gap-3">
          <span className="text-[13px] font-medium" style={{ color: analyzer.accent }}>
            Accepted input
          </span>
          <span className="font-mono text-[10px] text-ink-faint">10 MB maximum</span>
        </div>
        <p className="mt-1 text-[12px] leading-relaxed text-ink-dim">{analyzer.formats}</p>

        <div className="mt-4">
          <label htmlFor="upload-source-label" className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Company / data source <span className="normal-case tracking-normal">(optional)</span>
          </label>
          <input
            id="upload-source-label"
            value={sourceLabel}
            maxLength={160}
            onChange={(event) => setSourceLabel(event.target.value)}
            placeholder="e.g. Acme billing export (provided by you)"
            className="mt-2 w-full rounded-xl border border-line bg-void-3 px-3.5 py-2.5 text-[13px] text-ink outline-none placeholder:text-ink-faint focus:border-hub/60"
          />
          <p className="mt-1 text-[10px] text-ink-faint">This label is shown as user-provided; the app does not verify or infer a company.</p>
        </div>

        <label
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`mt-4 flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed px-5 py-6 text-center transition ${
            dragging ? "border-hub bg-hub/10" : "border-line bg-void-3/70 hover:border-ink-faint"
          }`}
        >
          <input ref={inputRef} type="file" accept={analyzer.accept} onChange={onFileChange} className="sr-only" />
          <span className="grid h-10 w-10 place-items-center rounded-full border border-line bg-void-2 text-lg text-ink-dim" aria-hidden="true">
            ↑
          </span>
          <span className="mt-3 text-[13px] font-medium text-ink">
            {file ? file.name : "Drop a file here, or browse"}
          </span>
          <span className="mt-1 text-[11px] text-ink-faint">
            {moduleId === "rag"
              ? "Extracted text stays in bounded API memory for this session; it is not persisted."
              : "The uploaded table is not saved; only a compact, temporary analysis note is indexed for session Q&A."}
          </span>
        </label>

        {moduleId === "rag" && (
          <div className="mt-4">
            <label htmlFor="rag-query" className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
              Question (optional)
            </label>
            <input
              id="rag-query"
              value={query}
              maxLength={2_000}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ask about the uploaded document…"
              className="mt-2 w-full rounded-xl border border-line bg-void-3 px-3.5 py-2.5 text-[13px] text-ink outline-none placeholder:text-ink-faint focus:border-hub/60"
            />
            {sessionId && (
              <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2">
                <p className="text-[10.5px] text-ink-faint">This browser session has an in-memory document index. It expires when the API restarts or evicts it.</p>
                <button
                  type="button"
                  onClick={() => {
                    const nextSession = createSessionId();
                    saveSessionId(nextSession);
                    saveSessionSources([]);
                    window.dispatchEvent(new CustomEvent("saros:session-updated", {
                      detail: { sessionId: nextSession, sources: [] },
                    }));
                    setSessionId(nextSession);
                    setFile(null);
                    setResult(null);
                    setError(null);
                    if (inputRef.current) inputRef.current.value = "";
                  }}
                  className="font-mono text-[10px] text-ink-faint underline underline-offset-2 hover:text-ink"
                >
                  Start new session
                </button>
              </div>
            )}
          </div>
        )}

        {error && (file || ragCanQuery ? (
          <div className="mt-3"><WakeRetry error={error} onRetry={() => void analyze()} busy={busy} label="Retry analysis" /></div>
        ) : (
          <p role="alert" className="mt-3 rounded-xl border border-churn/30 bg-churn/5 px-3 py-2.5 text-[12px] text-churn">{error}</p>
        ))}

        <button
          type="button"
          onClick={() => void analyze()}
          disabled={busy || (!file && !ragCanQuery)}
          className="mt-4 w-full rounded-xl bg-hub px-4 py-3 text-[13px] font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {busy ? "Analyzing your data…" : moduleId === "rag" && !file ? "Search this session" : "Analyze your data"}
        </button>
      </div>

      <div className="glass min-h-64 rounded-2xl p-5 sm:p-6" aria-live="polite">
        {!result && !busy && (
          <div className="flex h-full min-h-52 flex-col justify-center">
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">What you get</p>
            <h3 className="mt-3 text-xl font-semibold tracking-tight">A useful read on your file.</h3>
            <p className="mt-2 max-w-md text-[13px] leading-relaxed text-ink-dim">
              Get row counts, model metrics and a small set of scored examples using the existing body model. Uploads do not retrain or modify it.
            </p>
          </div>
        )}
        {busy && (
          <div className="flex min-h-52 flex-col items-center justify-center text-center">
            <span className="h-2 w-2 animate-ping rounded-full bg-hub" />
            <p className="mt-4 text-sm font-medium">The body is analyzing your data</p>
            <p className="mt-1 text-[12px] text-ink-faint">Large files can take a moment.</p>
          </div>
        )}
        {result && <AnalysisResult result={result} />}
      </div>
    </div>
  );
}

function predictionBasis(prediction: Prediction, moduleId: string) {
  const meta = prediction.meta;
  const num = (key: string, digits = 3) =>
    typeof meta[key] === "number" ? (meta[key] as number).toFixed(digits) : null;
  if (moduleId === "fraud") {
    return `Decisive model: ${String(meta.decisive_model ?? "not reported")} · supervised ${num("supervised_score") ?? "—"} · anomaly ${num("unsupervised_score") ?? "—"} · threshold ${num("threshold") ?? "—"}`;
  }
  if (moduleId === "churn") {
    return `Model source: ${String(meta.model_source ?? "existing churn model")} · threshold ${num("threshold") ?? "—"}`;
  }
  if (moduleId === "maintenance") {
    return `Engine ${String(meta.unit ?? "—")} at cycle ${String(meta.cycle ?? "—")} · predicted RUL ${num("rul_cycles", 1) ?? "—"} of ${String(meta.rul_cap ?? "—")} cycles · source ${String(meta.source ?? "not reported")}`;
  }
  if (moduleId === "pricing") {
    const interval = Array.isArray(meta.elasticity_ci) ? meta.elasticity_ci.map(String).join(" to ") : "not reported";
    const uplift = typeof meta.estimated_revenue_uplift === "number" ? `${(meta.estimated_revenue_uplift * 100).toFixed(1)}%` : "not reported";
    const range = Array.isArray(meta.observed_price_range) ? meta.observed_price_range.map((value) => `$${String(value)}`).join(" to ") : "not reported";
    return `SKU ${String(meta.sku ?? "—")} · elasticity ${num("elasticity") ?? "—"} (95% CI ${interval}) · R² ${num("r2") ?? "—"} · recommended price $${num("recommended_price", 2) ?? "—"} · estimated uplift ${uplift} · observed range ${range} · demand floor ${num("demand_floor", 2) ?? "—"}`;
  }
  return "";
}

function AnalysisResult({ result }: { result: UploadAnalysis }) {
  const scoreEntries = Object.entries(result.metrics.scores).slice(0, 4);
  const firstPrediction = result.predictions[0];
  const featureNames = result.module_id === "rag"
    ? []
    : result.module_id === "pricing"
      ? ["sku", "price", "demand"]
      : firstPrediction ? Object.keys(firstPrediction.inputs) : [];
  const methodBasis = result.module_id === "rag"
    ? "Heading-aware document chunks are ranked with lexical TF-IDF; matching passages retain their original file and heading for citation."
    : result.metrics.notes[0] ?? "The existing body analyzer scored the uploaded data.";
  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">Analysis complete</p>
          <h3 className="mt-1 text-lg font-semibold">{result.filename ?? "Session corpus"}</h3>
          <p className="mt-1 text-[10.5px] text-ink-faint">
            Company / source: {result.source_label ? `${result.source_label} (user-provided)` : "not specified; none inferred"}
          </p>
        </div>
        <span className="rounded-full border border-line bg-void-3 px-3 py-1 font-mono text-[10px] text-ink-dim">
          {result.n_rows.toLocaleString()} {result.module_id === "rag" ? "documents" : "rows"}
        </span>
      </div>

      <div className="mt-3 rounded-xl border border-line bg-void-3/70 p-3">
        <p className="font-mono text-[9px] uppercase tracking-[0.15em] text-ink-faint">Data & model basis</p>
        <p className="mt-1.5 text-[11px] leading-relaxed text-ink-dim">{methodBasis}</p>
        {featureNames.length > 0 && (
          <p className="mt-1.5 text-[10px] leading-relaxed text-ink-faint">
            Fields used: {featureNames.join(", ")}
          </p>
        )}
        {(result.module_id === "churn" || result.module_id === "fraud") && (
          <p className="mt-1 text-[10px] text-ink-faint">Operating threshold: {result.metrics.threshold.toFixed(3)}</p>
        )}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        {scoreEntries.map(([name, value]) => (
          <div key={name} className="rounded-xl border border-line bg-void-3/70 px-3 py-2.5">
            <div className="font-mono text-[9px] uppercase tracking-wider text-ink-faint">{name.replaceAll("_", " ")}</div>
            <div className="mt-1 text-base font-semibold tabular-nums">
              {Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3)}
            </div>
          </div>
        ))}
      </div>

      {result.predictions.length > 0 && (
        <div className="mt-4 space-y-2">
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">Scored examples</p>
          {result.predictions.slice(0, 3).map((prediction, index) => (
            <div key={`${prediction.module_id}-${index}`} className="rounded-xl border border-line bg-void-3/70 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[12px] font-medium">
                  {prediction.label}
                  {prediction.meta.row_index !== undefined && (
                    <span className="ml-1.5 font-mono text-[9px] text-ink-faint">row {String(prediction.meta.row_index)}</span>
                  )}
                </span>
                <span className="font-mono text-[10px] tabular-nums text-ink-dim">
                  {result.module_id === "pricing" ? "elasticity index" : result.module_id === "maintenance" ? "urgency" : result.module_id === "rag" ? "match" : "risk score"} {prediction.score.toFixed(3)}
                </span>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-ink-dim">{prediction.decision}</p>
              {predictionBasis(prediction, result.module_id) && (
                <p className="mt-1.5 text-[10px] leading-relaxed text-ink-faint">{predictionBasis(prediction, result.module_id)}</p>
              )}
              {prediction.explanation?.summary && (
                <p className="mt-1.5 text-[10px] leading-relaxed text-ink-faint">Why: {prediction.explanation.summary}</p>
              )}
              {prediction.factors.length > 0 && (
                <p className="mt-1.5 text-[10px] leading-relaxed text-ink-faint">
                  Local factors: {prediction.factors.slice(0, 3).map((factor) => `${factor.feature}=${factor.value} (${factor.attribution >= 0 ? "+" : ""}${factor.attribution})`).join(" · ")}
                </p>
              )}
              {Array.isArray(prediction.meta.hits) && prediction.meta.hits.length > 0 && (
                <p className="mt-2 truncate font-mono text-[10px] text-ink-faint">
                  Source: {(() => {
                    const source = String((prediction.meta.hits[0] as { source?: string }).source ?? "uploaded corpus");
                    return source.lastIndexOf(".") > source.lastIndexOf("/") ? source : `${source}.md`;
                  })()}
                </p>
              )}
            </div>
          ))}
          {result.predictions_truncated && <p className="text-[10px] text-ink-faint">Large uploads return a maximum of 100 scored examples.</p>}
        </div>
      )}

      <ul className="mt-4 space-y-1 border-t border-line pt-3">
        {result.metrics.notes.slice(0, 3).map((note) => (
          <li key={note} className="text-[10.5px] leading-relaxed text-ink-faint">{note}</li>
        ))}
      </ul>
      {result.session_sources?.length > 0 && (
        <p className="mt-3 text-[10px] leading-relaxed text-ink-faint">
          Temporary session Q&A sources: {result.session_sources.slice(0, 4).join(", ")}
          {result.session_sources.length > 4 ? ` +${result.session_sources.length - 4} more` : ""}
        </p>
      )}
      {result.session_id && (
        <button
          type="button"
          onClick={() => {
            const question = `Use the uploaded analysis report for ${result.filename ?? "my upload"}. Explain the main result, which model and fields were used, why the highest-scored rows received their scores, and any caveats. Search my active upload session and cite the exact source file and heading.`;
            window.dispatchEvent(new CustomEvent("saros:prefill-agent", { detail: question }));
            document.getElementById("agent")?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
          className="mt-4 w-full rounded-xl border border-hub/40 bg-hub/10 px-3 py-2.5 text-[11px] font-medium text-ink transition hover:bg-hub/20"
        >
          Ask the hub agent about this upload →
        </button>
      )}
    </div>
  );
}
