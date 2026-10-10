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
import { useEffect, useState } from "react";

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
}

interface Sources {
  n_documents: number;
  n_passages: number;
  sources: string[];
}

const ACCENT = "#0284c7";

const SUGGESTIONS = [
  "What discount can an agent approve?",
  "What is the refund window?",
  "How do I escalate a disputed refund?",
  "What counts as card testing?",
  "What is the uptime commitment?",
];

function verdictStyle(grounding: string) {
  if (grounding === "Well grounded") return { bg: "#f0fdf4", fg: "#166534", border: "#bbf7d0" };
  if (grounding === "Weak grounding") return { bg: "#fffbeb", fg: "#92400e", border: "#fde68a" };
  return { bg: "#fef2f2", fg: "#991b1b", border: "#fecaca" };
}

export function KnowledgeSearch() {
  const [query, setQuery] = useState("");
  const [res, setRes] = useState<SearchResult | null>(null);
  const [sources, setSources] = useState<Sources | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    clientFetch<{ result: Sources }>("/api/v1/agent/tools/list_knowledge_sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: {} }),
    })
      .then((r) => setSources(r.result))
      .catch(() => setSources(null));
  }, []);

  async function run(q: string) {
    const text = q.trim();
    if (!text || busy) return;
    setQuery(text);
    setBusy(true);
    setError(null);
    try {
      const r = await clientFetch<{ result: SearchResult }>("/api/v1/agent/tools/search_knowledge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arguments: { query: text, k: 5 } }),
      });
      setRes(r.result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const v = res ? verdictStyle(res.grounding) : null;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
      <div className="rounded-2xl border border-[#e7e0d5] bg-white p-6 shadow-sm">
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
            className="flex-1 rounded-xl border border-[#e7e0d5] px-3.5 py-2.5 text-[13px] outline-none focus:border-[#0284c7]"
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
                className="rounded-full border border-[#e7e0d5] px-2.5 py-1 text-[11.5px] text-[#57534e] transition hover:border-[#0284c7] hover:text-[#0284c7]"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-xl border border-[#fecaca] bg-[#fef2f2] p-4 text-[12.5px] text-[#991b1b]">
            {error}
          </div>
        )}

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
            </div>

            <div className="mt-4 space-y-2.5">
              {res.results.map((h, i) => (
                <motion.article
                  key={`${h.source}-${h.heading}-${i}`}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="rounded-xl border border-[#f5f5f4] p-4"
                >
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div className="flex items-baseline gap-2">
                      <span className="font-mono text-[10.5px] text-[#a8a29e]">{h.source}.md</span>
                      <span className="text-[13px] font-medium">{h.heading}</span>
                    </div>
                    <span className="font-mono text-[11px] tabular-nums text-[#78716c]">
                      {h.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="mt-2 text-[12.5px] leading-relaxed text-[#57534e]">{h.snippet}</p>
                </motion.article>
              ))}
            </div>

            {res.results.length === 0 && (
              <p className="mt-4 rounded-xl border border-[#fecaca] bg-[#fef2f2] p-4 text-[12.5px] text-[#991b1b]">
                Nothing in the corpus matches. The honest answer is &ldquo;I don&rsquo;t know&rdquo;,
                not a plausible guess.
              </p>
            )}
          </div>
        )}
      </div>

      <div className="space-y-3">
        {sources && (
          <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">
              Corpus
            </div>
            <div className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">
              {sources.n_documents}
            </div>
            <div className="text-[12px] text-[#57534e]">
              documents · {sources.n_passages} passages
            </div>
            <ul className="mt-3 space-y-1 border-t border-[#f5f5f4] pt-3">
              {sources.sources.map((s) => (
                <li key={s} className="font-mono text-[11px] text-[#78716c]">
                  {s}.md
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">
            Grounding rule
          </div>
          <p className="mt-2 text-[11.5px] leading-relaxed text-[#57534e]">
            A verdict rests on how many <em>substantive</em> query terms appear in the top
            passage, not on the cosine. On a 25-passage corpus TF-IDF tops out near 0.22 even for
            obviously-relevant questions, so a similarity floor would reject everything.
          </p>
          <p className="mt-2 border-t border-[#f5f5f4] pt-2 text-[11px] leading-relaxed text-[#a8a29e]">
            One shared token is not grounding: &ldquo;airspeed velocity of an unladen swallow&rdquo;
            matches the fraud protocol on &ldquo;velocity&rdquo; alone.
          </p>
        </div>
      </div>
    </div>
  );
}
