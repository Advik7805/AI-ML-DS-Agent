"use client";

/**
 * Analyze your own data — the hub's upload studio.
 *
 * Three steps, in the order a person actually thinks: pick the body whose model you want, drop a
 * file, read what it found. The mapping panel is not decoration: it names which of your columns
 * were used, which were ignored and — when the file is rejected — exactly which columns the body
 * needed. A refusal that says "invalid file" is the reason most upload buttons are useless.
 *
 * Then the part that makes it a product rather than a file input: the session id travels with
 * every request, so the hub agent can be asked about the file you just dropped, in the same
 * conversation, with no re-upload.
 */

import { useCallback, useMemo, useRef, useState } from "react";

import { readSessionId, useSessionId } from "@/lib/session";
import type { DatasetCard, MetricsReport, ModuleManifest, Prediction } from "@/lib/types";

interface ColumnsReport {
  expected: string[];
  matched: string[];
  missing: string[];
  ignored: string[];
  recognised_optional: string[];
}

interface Analysis {
  session_id: string;
  module_id: string;
  filename: string;
  size_bytes: number;
  kind: string;
  rows: number | null;
  columns_report: ColumnsReport;
  threshold: number;
  metrics: MetricsReport | null;
  predictions: Prediction[];
  dataset: DatasetCard | null;
  preview: Record<string, string>[];
  notes: string[];
  extra: Record<string, unknown>;
}

interface Rejection {
  message: string;
  expected_columns?: string[];
  found_columns?: string[];
  hint?: string;
}


export function UploadStudio({ modules }: { modules: ModuleManifest[] }) {
  const session = useSessionId();
  const [moduleId, setModuleId] = useState<string>(modules[0]?.id ?? "churn");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [rejection, setRejection] = useState<Rejection | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selected = useMemo(
    () => modules.find((m) => m.id === moduleId) ?? modules[0],
    [modules, moduleId],
  );
  const accent = selected?.accent ?? "#7c5cff";

  const reset = () => {
    setAnalysis(null);
    setRejection(null);
    setFailure(null);
    setAnswer(null);
  };

  const upload = useCallback(
    async (file: File) => {
      reset();
      if (file.size > 10 * 1024 * 1024) {
        setRejection({
          message: `${file.name} is ${(file.size / 1024 / 1024).toFixed(1)} MB, over the 10 MB cap.`,
          hint: "Send a sample — a few thousand rows shows how the model behaves.",
        });
        return;
      }
      setBusy(true);
      setProgress(`Uploading ${file.name}…`);
      const id = session || readSessionId();
      const form = new FormData();
      form.append("file", file);
      form.append("session_id", id);
      try {
        const res = await fetch(`/api/v1/modules/${moduleId}/analyze`, {
          method: "POST",
          headers: { "x-saros-session": id },
          body: form,
        });
        setProgress(res.status === 503 || res.status === 502 ? "The API is waking up…" : null);
        const payload = (await res.json()) as Analysis | { detail: Rejection | string };
        if (!res.ok) {
          const detail = (payload as { detail: Rejection | string }).detail;
          if (typeof detail === "string") setRejection({ message: detail });
          else setRejection(detail);
          return;
        }
        setAnalysis(payload as Analysis);
      } catch (e) {
        setFailure(
          e instanceof Error
            ? `${e.message} — the API may be waking up (free tiers sleep after 15 min idle). Try again in a few seconds.`
            : String(e),
        );
      } finally {
        setBusy(false);
        setProgress(null);
      }
    },
    [moduleId, session],
  );

  const ask = useCallback(
    async (text: string) => {
      if (!text.trim() || !analysis) return;
      setAsking(true);
      setAnswer(null);
      try {
        const res = await fetch("/api/v1/agent/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: session || readSessionId() }),
        });
        const payload = (await res.json()) as { answer?: string; error?: string | null; detail?: string };
        setAnswer(payload.answer || payload.detail || payload.error || "No answer returned.");
      } catch (e) {
        setAnswer(`Agent unreachable: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setAsking(false);
      }
    },
    [analysis, session],
  );

  const suggestions = useMemo(() => {
    if (!analysis) return [];
    if (analysis.kind === "document") {
      return [
        `Summarise the key policy points in ${analysis.filename}.`,
        `What does ${analysis.filename} say about deadlines or windows?`,
        `Which exceptions does ${analysis.filename} list?`,
      ];
    }
    return [
      `What did the ${selected?.title ?? analysis.module_id} model find in ${analysis.filename}?`,
      `Which rows in ${analysis.filename} look riskiest, and why?`,
      `What should I check before trusting these results?`,
    ];
  }, [analysis, selected]);

  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-semibold tracking-tight">Analyze your data</h3>
          <p className="mt-1 max-w-2xl text-[12.5px] leading-relaxed text-ink-dim">
            Drop your own file and the body you choose runs it through the same model that scores
            the shipped tenant data. Nothing is written to disk, and the agent can answer
            questions about the file you just dropped.
          </p>
        </div>
        {session && (
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            session {session.slice(0, 12)}…
          </span>
        )}
      </div>

      {/* --- body picker ---------------------------------------------------- */}
      <div className="mt-4 flex flex-wrap gap-1.5">
        {modules.map((m) => (
          <button
            key={m.id}
            onClick={() => {
              setModuleId(m.id);
              reset();
            }}
            className="rounded-full border px-3 py-1 text-[11.5px] transition"
            style={
              m.id === moduleId
                ? { borderColor: `${m.accent}88`, background: `${m.accent}22`, color: "#e8eaf2" }
                : { borderColor: "var(--color-line)", color: "var(--color-ink-dim)" }
            }
          >
            {m.title}
          </button>
        ))}
      </div>

      {selected?.accepts && (
        <p className="mt-3 rounded-xl border border-line bg-void-2/60 px-3.5 py-2.5 text-[11.5px] leading-relaxed text-ink-dim">
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            {selected.title} accepts ·{" "}
          </span>
          {selected.accepts}
        </p>
      )}

      {/* --- dropzone -------------------------------------------------------- */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) void upload(file);
        }}
        onClick={() => inputRef.current?.click()}
        className="mt-4 cursor-pointer rounded-2xl border border-dashed px-6 py-8 text-center transition"
        style={{
          borderColor: dragging ? accent : "var(--color-line)",
          background: dragging ? `${accent}12` : "rgba(255,255,255,0.02)",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
            e.target.value = "";
          }}
        />
        <div className="text-[13px] font-medium">
          {busy ? "Analyzing…" : dragging ? "Drop to analyze" : "Drag a file here, or click to choose"}
        </div>
        <div className="mt-1.5 font-mono text-[10.5px] uppercase tracking-wider text-ink-faint">
          csv · tsv · txt · md · pdf · 10 MB max · in-memory only
        </div>
        {(progress || busy) && (
          <div className="mt-3 flex items-center justify-center gap-2 text-[11.5px] text-ink-dim">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: accent }} />
            {progress ?? "Scoring your rows — larger files take a few seconds."}
          </div>
        )}
      </div>

      {/* --- rejection: name the columns ------------------------------------ */}
      {rejection && (
        <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <div className="text-[12.5px] font-medium text-red-100">{rejection.message}</div>
          {rejection.found_columns && rejection.found_columns.length > 0 && (
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <ColumnList title="your columns" columns={rejection.found_columns} tone="#fca5a5" />
              {rejection.expected_columns && (
                <ColumnList title="what this body needs" columns={rejection.expected_columns} tone="#6ee7b7" />
              )}
            </div>
          )}
          {rejection.hint && <p className="mt-3 text-[11.5px] leading-relaxed text-red-200/80">{rejection.hint}</p>}
        </div>
      )}

      {failure && (
        <div className="mt-4 rounded-xl border border-churn/40 bg-churn/10 p-4 text-[12.5px] text-ink-dim">
          {failure}
        </div>
      )}

      {/* --- results --------------------------------------------------------- */}
      {analysis && (
        <div className="mt-5 space-y-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[13px] font-medium">
              {analysis.filename}
              <span className="ml-2 font-mono text-[10.5px] text-ink-faint">
                {analysis.rows !== null ? `${analysis.rows.toLocaleString()} ${analysis.kind === "document" ? "chunks" : "rows"}` : ""}
                {analysis.size_bytes ? ` · ${(analysis.size_bytes / 1024).toFixed(0)} KB` : ""}
              </span>
            </div>
            {analysis.dataset && (
              <span className="rounded-full border border-emerald-500/40 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-emerald-300">
                processed in memory · not stored
              </span>
            )}
          </div>

          <div className="grid gap-3 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
            <div className="rounded-xl border border-line bg-void-2/60 p-4">
              <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                column mapping
              </div>
              <div className="mt-3 space-y-3">
                {analysis.columns_report.matched.length > 0 && (
                  <ColumnList
                    title={`used by the model (${analysis.columns_report.matched.length})`}
                    columns={analysis.columns_report.matched}
                    tone="#6ee7b7"
                  />
                )}
                {analysis.columns_report.ignored.length > 0 && (
                  <ColumnList
                    title={`present but ignored (${analysis.columns_report.ignored.length})`}
                    columns={analysis.columns_report.ignored}
                    tone="var(--color-ink-faint)"
                  />
                )}
                {analysis.columns_report.matched.length === 0 && analysis.kind === "document" && (
                  <p className="text-[11.5px] text-ink-dim">
                    A document, not a table: chunked heading-wise and added to this session&rsquo;s
                    knowledge overlay.
                  </p>
                )}
              </div>
            </div>

            <div className="space-y-3">
              {analysis.metrics && (
                <div className="rounded-xl border border-line bg-void-2/60 p-4">
                  <div className="flex items-baseline justify-between">
                    <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                      {analysis.metrics.scores?.pr_auc !== undefined
                        ? "metrics against your labels"
                        : "score distribution on your file"}
                    </div>
                    <div className="font-mono text-[10px] text-ink-faint">
                      n={analysis.metrics.n_samples} · threshold {analysis.threshold.toFixed(2)}
                    </div>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
                    {Object.entries(analysis.metrics.scores).slice(0, 9).map(([k, v]) => (
                      <div key={k} className="rounded-lg bg-white/[0.03] px-2.5 py-2">
                        <div className="font-mono text-[9.5px] uppercase tracking-wider text-ink-faint">
                          {k.replace(/_/g, " ")}
                        </div>
                        <div className="mt-0.5 text-[14px] font-semibold tabular-nums">
                          {typeof v === "number" ? v.toFixed(3) : String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                  {analysis.metrics.notes.length > 0 && (
                    <ul className="mt-3 space-y-1 border-t border-line pt-3">
                      {analysis.metrics.notes.slice(0, 4).map((n) => (
                        <li key={n} className="text-[11px] leading-relaxed text-ink-faint">
                          — {n}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {analysis.predictions.length > 0 && (
                <div className="rounded-xl border border-line bg-void-2/60 p-4">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                    top {Math.min(5, analysis.predictions.length)} of {analysis.predictions.length} scored rows
                  </div>
                  <ul className="mt-3 space-y-2">
                    {analysis.predictions.slice(0, 5).map((p, i) => (
                      <li key={i} className="rounded-lg bg-white/[0.03] px-3 py-2">
                        <div className="flex items-baseline justify-between gap-3">
                          <span className="text-[12.5px] font-medium" style={{ color: accent }}>
                            {p.label}
                          </span>
                          <span className="font-mono text-[12px] tabular-nums">{p.score.toFixed(3)}</span>
                        </div>
                        <div className="mt-1 text-[11.5px] leading-relaxed text-ink-dim">{p.decision}</div>
                        {p.factors.length > 0 && (
                          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5">
                            {p.factors.slice(0, 4).map((f) => (
                              <span key={f.feature} className="font-mono text-[10.5px] text-ink-faint">
                                {f.feature} {f.attribution > 0 ? "+" : ""}
                                {f.attribution.toFixed(2)}
                              </span>
                            ))}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {analysis.preview.length > 0 && analysis.kind === "document" && (
                <div className="rounded-xl border border-line bg-void-2/60 p-4">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
                    indexed sections
                  </div>
                  <ul className="mt-2 space-y-2">
                    {analysis.preview.map((row, i) => (
                      <li key={i} className="text-[11.5px] leading-relaxed text-ink-dim">
                        <span className="font-mono text-[11px] text-ink">{row.heading}</span> — {row.text}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* --- ask about it ------------------------------------------------ */}
          <div className="rounded-xl border p-4" style={{ borderColor: `${accent}44` }}>
            <div className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
              ask the hub agent about this file
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void ask(question);
              }}
              className="mt-2.5 flex flex-wrap gap-2"
            >
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={`e.g. what did you find in ${analysis.filename}?`}
                className="min-w-[240px] flex-1 rounded-xl border border-line bg-void-2/60 px-3.5 py-2.5 text-[13px] text-ink outline-none placeholder:text-ink-faint focus:border-hub/60"
              />
              <button
                type="submit"
                disabled={asking || !question.trim()}
                className="rounded-xl px-4 py-2.5 text-[13px] font-medium text-white transition disabled:opacity-40"
                style={{ background: accent }}
              >
                {asking ? "Thinking…" : "Ask"}
              </button>
            </form>
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setQuestion(s);
                    void ask(s);
                  }}
                  className="rounded-full border border-line px-2.5 py-1 text-[11px] text-ink-dim transition hover:border-hub/50 hover:text-ink"
                >
                  {s}
                </button>
              ))}
            </div>
            {asking && (
              <div className="mt-3 flex items-center gap-2 text-[11.5px] text-ink-faint">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full" style={{ background: accent }} />
                reading your file&rsquo;s profile and retrieving matching passages…
              </div>
            )}
            {answer && (
              <div className="mt-3 whitespace-pre-wrap rounded-lg bg-white/[0.04] px-3.5 py-2.5 text-[12.5px] leading-relaxed text-ink ring-1 ring-white/10">
                {answer}
              </div>
            )}
          </div>

          {analysis.notes.length > 0 && (
            <ul className="space-y-1">
              {analysis.notes.slice(0, 5).map((n) => (
                <li key={n} className="text-[11px] leading-relaxed text-ink-faint">
                  — {n}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function ColumnList({ title, columns, tone }: { title: string; columns: string[]; tone: string }) {
  return (
    <div>
      <div className="font-mono text-[9.5px] uppercase tracking-wider" style={{ color: tone }}>
        {title}
      </div>
      <div className="mt-1 flex flex-wrap gap-1">
        {columns.slice(0, 24).map((c) => (
          <span
            key={c}
            className="rounded border border-line bg-white/[0.03] px-1.5 py-0.5 font-mono text-[10px] text-ink-dim"
          >
            {c}
          </span>
        ))}
        {columns.length > 24 && (
          <span className="font-mono text-[10px] text-ink-faint">+{columns.length - 24} more</span>
        )}
      </div>
    </div>
  );
}
