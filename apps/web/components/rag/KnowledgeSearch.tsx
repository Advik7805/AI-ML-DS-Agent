"use client";

/**
 * Grounded knowledge search.
 *
 * This is a retrieval console, not a chatbot: it shows the passages, their scores and the
 * grounding verdict, and refuses to dress a weak match up as an answer. The verdict is the same
 * one `predict()` returns, because the body computes it in one place — a storefront and an agent
 * that disagree about whether something was grounded is exactly how RAG systems earn a bad name.
 */

import { motion } from "motion/react";
import { useCallback, useEffect, useState } from "react";

import { WakeRetry } from "@/components/body/WakeRetry";
import { clientFetch } from "@/lib/api";

interface Hit {
  source: string;
  heading: string;
  score: number;
  snippet: string;
}

interface SearchResult {
  query: string;
  n_results: number;
  grounding: string;
  matched_terms: number;
  decision: string;
  results: Hit[];
  citation: string;
  session_sources?: string[];
  session_scoped?: boolean;
}

/** Provenance of the loaded corpus. Mirrors `corpus_status()` in modules/rag/service.py. */
interface CorpusStatus {
  external_backend: string;
  external_live: boolean;
  n_external_passages: number;
  external_error?: string;
  warning?: string;
}

interface Sources {
  n_documents: number;
  n_passages: number;
  sources: string[];
  session_sources?: string[];
  session_requested?: boolean;
  /** Present once the API reports it. Optional so an older backend does not break the board. */
  corpus?: CorpusStatus;
}

const ACCENT = "#38bdf8";

const SUGGESTIONS = [
  "What discount can an agent approve?",
  "What is the refund window?",
  "How do I escalate a disputed refund?",
  "What counts as card testing?",
  "What is the uptime commitment?",
];

function verdictStyle(_grounding: string) {
  return {
    bg: "var(--color-void-3)",
    fg: "var(--color-ink)",
    border: "var(--color-line)",
  };
}

function sourceFilename(source: string) {
  return source.lastIndexOf(".") > source.lastIndexOf("/") ? source : `${source}.md`;
}

export function KnowledgeSearch() {
  const [query, setQuery] = useState("");
  const [res, setRes] = useState<SearchResult | null>(null);
  const [sources, setSources] = useState<Sources | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [sessionSources, setSessionSources] = useState<string[]>([]);
  const [sourcesBusy, setSourcesBusy] = useState(true);
  const [sourcesError, setSourcesError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSources = useCallback(async () => {
    setSourcesBusy(true);
    setSourcesError(null);
    let activeSession = "";
    try {
      activeSession = window.localStorage.getItem("saros-rag-session") ?? "";
      setSessionId(activeSession);
      const savedSources = JSON.parse(window.localStorage.getItem("saros-session-sources") ?? "[]") as unknown;
      setSessionSources(Array.isArray(savedSources) ? savedSources.filter((item): item is string => typeof item === "string") : []);
    } catch {
      setSessionSources([]);
    }
    try {
      const response = await clientFetch<{ result: Sources }>("/api/v1/agent/tools/list_knowledge_sources", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arguments: activeSession ? { session_id: activeSession } : {} }),
      });
      setSources(response.result);
      if (response.result.session_sources) {
        setSessionSources(response.result.session_sources);
        try {
          window.localStorage.setItem("saros-session-sources", JSON.stringify(response.result.session_sources));
        } catch {
          /* session source names are just a display convenience */
        }
      }
    } catch (reason) {
      setSourcesError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSourcesBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadSources();
  }, [loadSources]);

  async function run(q: string) {
    const text = q.trim();
    if (!text || busy) return;
    setQuery(text);
    setBusy(true);
    setError(null);
    try {
      const activeSession = sessionId || window.localStorage.getItem("saros-rag-session") || "";
      const r = await clientFetch<{ result: SearchResult }>("/api/v1/agent/tools/search_knowledge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arguments: { query: text, k: 5, ...(activeSession ? { session_id: activeSession } : {}) },
        }),
      });
      setRes(r.result);
      if (r.result.session_sources) {
        setSessionSources(r.result.session_sources);
        try {
          window.localStorage.setItem("saros-session-sources", JSON.stringify(r.result.session_sources));
        } catch {
          /* session source names are just a display convenience */
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const v = res ? verdictStyle(res.grounding) : null;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <div className="rounded-2xl border border-line bg-void-2 p-6 shadow-sm">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run(query);
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask the corpus…"
            className="flex-1 rounded-xl border border-line bg-void-3 px-3.5 py-2.5 text-[13px] text-ink outline-none placeholder:text-ink-faint focus:border-ink-dim"
          />
          <button
            type="submit"
            disabled={busy || !query.trim()}
            className="rounded-xl px-4 py-2.5 text-[13px] font-medium text-white transition hover:brightness-110 disabled:opacity-40"
            style={{ background: ACCENT }}
          >
            {busy ? "Searching…" : "Search"}
          </button>
        </form>

        {!res && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => void run(s)}
                className="rounded-full border border-line px-2.5 py-1 text-[11.5px] text-ink-dim transition hover:border-ink-dim hover:text-ink"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {busy && <p role="status" className="mt-3 text-[11px] text-ink-faint">Searching the index and ranking source passages…</p>}
        {error && <div className="mt-4"><WakeRetry error={error} onRetry={() => void run(query)} busy={busy} label="Retry search" /></div>}

        {res && v && (
          <div className="mt-5">
            <div
              className="rounded-xl border p-4"
              style={{ background: v.bg, borderColor: v.border }}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-[13px] font-semibold" style={{ color: v.fg }}>
                  {res.grounding}
                </span>
                <span className="font-mono text-[10.5px]" style={{ color: v.fg }}>
                  {res.matched_terms} substantive terms matched · best score{" "}
                  {res.results[0]?.score.toFixed(3) ?? "0.000"}
                </span>
              </div>
              <p className="mt-1.5 text-[12px] leading-relaxed" style={{ color: v.fg }}>
                {res.decision}
              </p>
              {res.session_sources && res.session_sources.length > 0 && (
                <p className="mt-2 border-t border-line pt-2 text-[10px] leading-relaxed text-ink-faint">
                  Searched this browser session's uploads: {res.session_sources.join(", ")}
                </p>
              )}
            </div>

            <div className="mt-4 space-y-2.5">
              {res.results.map((h, i) => (
                <motion.article
                  key={`${h.source}-${h.heading}-${i}`}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="rounded-xl border border-line p-4"
                >
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div className="flex items-baseline gap-2">
                      <span className="font-mono text-[10.5px] text-ink-faint">{sourceFilename(h.source)}</span>
                      <span className="text-[13px] font-medium">{h.heading}</span>
                    </div>
                    <span className="font-mono text-[11px] tabular-nums text-ink-dim">
                      {h.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="mt-2 text-[12.5px] leading-relaxed text-ink-dim">{h.snippet}</p>
                </motion.article>
              ))}
            </div>

            {res.results.length === 0 && (
              <p className="mt-4 rounded-xl border border-line bg-void-3 p-4 text-[12.5px] text-ink-dim">
                Nothing in the corpus matches. The honest answer is &ldquo;I don&rsquo;t know&rdquo;,
                not a plausible guess.
              </p>
            )}
          </div>
        )}
      </div>

      <div className="space-y-3">
        {sourcesError && <WakeRetry error={`Knowledge sources unavailable: ${sourcesError}`} onRetry={() => void loadSources()} busy={sourcesBusy} label="Retry sources" />}
        {sourcesBusy && !sources && <p role="status" className="rounded-xl border border-line bg-void-2 p-4 text-[11px] text-ink-faint">Waking the API and loading indexed sources…</p>}
        {sources && (
          <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
              Corpus
            </div>
            <div className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">
              {sources.n_documents}
            </div>
            <div className="text-[12px] text-ink-dim">
              documents · {sources.n_passages} passages
            </div>
            <ul className="mt-3 space-y-1 border-t border-line pt-3">
              {sources.sources.map((s) => (
                <li key={s} className="font-mono text-[11px] text-ink-dim">{sourceFilename(s)}</li>
              ))}
            </ul>

            <div className="mt-3 border-t border-line pt-3">
              <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">This browser session</div>
              {sessionSources.length > 0 ? (
                <ul className="mt-2 space-y-1">
                  {sessionSources.map((source) => (
                    <li key={source} className="font-mono text-[10px] text-ink-dim">{source}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-1 text-[10px] leading-relaxed text-ink-faint">
                  No upload report is indexed here yet. Upload a supported file from the hub to make it searchable by the agent.
                </p>
              )}
            </div>

            {sources.corpus && (
              <div className="mt-3 border-t border-line pt-3">
                <div className="flex items-center gap-1.5">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      sources.corpus.external_live ? "bg-ink-dim" : "bg-line"
                    }`}
                  />
                  <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
                    {sources.corpus.external_live
                      ? `live · ${sources.corpus.n_external_passages} extra passages`
                      : "in-repo corpus only"}
                  </span>
                </div>
                {/* An external corpus that failed to load must be visible. Without this a
                    Supabase outage is indistinguishable from a knowledge base that genuinely
                    has six documents — and the answers stay confident either way. */}
                {sources.corpus.warning && (
                  <p className="mt-2 rounded-md bg-void-3 p-2 text-[10.5px] leading-relaxed text-ink-dim">
                    {sources.corpus.warning}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        <div className="rounded-2xl border border-line bg-void-2 p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-faint">
            Grounding rule
          </div>
          <p className="mt-2 text-[11.5px] leading-relaxed text-ink-dim">
            A verdict rests on how many <em>substantive</em> query terms appear in the top
            passage, not on the cosine. On a 25-passage corpus TF-IDF tops out near 0.22 even for
            obviously-relevant questions, so a similarity floor would reject everything.
          </p>
          <p className="mt-2 border-t border-line pt-2 text-[11px] leading-relaxed text-ink-faint">
            One shared token is not grounding: &ldquo;airspeed velocity of an unladen swallow&rdquo;
            matches the fraud protocol on &ldquo;velocity&rdquo; alone.
          </p>
        </div>
      </div>
    </div>
  );
}
