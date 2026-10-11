"use client";

/**
 * Data acquisition console — plain HTTP, Tavily web search, and Tavily extraction.
 *
 * Three backends with genuinely different trade-offs, shown side by side rather than hidden
 * behind one button: a plain GET is free but cannot render JavaScript; Tavily renders JS and
 * costs credits. Every result states which backend produced it, and mock output is labelled in
 * the UI as well as in the payload — an operator who cannot tell simulated data from real data
 * will eventually ship a dashboard built on placeholders.
 */

import { motion } from "motion/react";
import { useEffect, useState } from "react";

import { clientFetch } from "@/lib/api";

interface ScrapeResult {
  url: string;
  status: number;
  bytes: number;
  fetcher: string;
  markdown?: string;
  error?: string;
}

interface SearchHit {
  title: string;
  url: string;
  content: string;
  score: number;
}

interface SearchResult {
  query: string;
  backend: string;
  live: boolean;
  answer?: string;
  results: SearchHit[];
  note?: string;
  error?: string;
}

interface ExtractResult {
  urls: string[];
  backend: string;
  live: boolean;
  results: Array<{ url: string; raw_content: string }>;
  error?: string;
}

interface Caps {
  live: boolean;
  allowlist: string[];
  fetcher: string;
  max_bytes: number;
  tavily: {
    live: boolean;
    has_key: boolean;
    key_env: string;
    enable_env: string;
    free_tier: string;
    warning: string | null;
  };
}

const ACCENT = "#ea580c";

type Tab = "fetch" | "search" | "extract";

async function call<T>(tool: string, args: Record<string, unknown>): Promise<T> {
  const r = await clientFetch<{ result: T }>(`/api/v1/agent/tools/${tool}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arguments: args }),
  });
  return r.result;
}

export function AcquisitionConsole() {
  const [tab, setTab] = useState<Tab>("fetch");
  const [caps, setCaps] = useState<Caps | null>(null);
  const [url, setUrl] = useState("https://example.com/pricing");
  const [query, setQuery] = useState("predictive maintenance benchmarks 2026");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scrape, setScrape] = useState<ScrapeResult | null>(null);
  const [search, setSearch] = useState<SearchResult | null>(null);
  const [extract, setExtract] = useState<ExtractResult | null>(null);

  useEffect(() => {
    call<Caps>("scraper_capabilities", {}).then(setCaps).catch(() => setCaps(null));
  }, []);

  async function runFetch() {
    setBusy(true); setError(null); setScrape(null);
    try { setScrape(await call<ScrapeResult>("scrape_url", { url })); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  async function runSearch() {
    setBusy(true); setError(null); setSearch(null);
    try { setSearch(await call<SearchResult>("web_search", { query, max_results: 5 })); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  async function runExtract() {
    setBusy(true); setError(null); setExtract(null);
    try {
      const urls = url.split(/[\s,]+/).filter(Boolean).slice(0, 5);
      setExtract(await call<ExtractResult>("extract_url", { urls }));
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
      <div className="rounded-2xl border border-[#e7e0d5] bg-white p-6 shadow-sm">
        <div className="flex gap-1 rounded-xl bg-[#f5f5f4] p-1">
          {([
            ["fetch", "Plain GET"],
            ["search", "Web search"],
            ["extract", "Extract text"],
          ] as [Tab, string][]).map(([id, label]) => (
            <button
              key={id}
              onClick={() => { setTab(id); setError(null); }}
              className={`flex-1 rounded-lg px-3 py-2 text-[12.5px] font-medium transition ${
                tab === id ? "bg-white shadow-sm text-[#1c1917]" : "text-[#78716c]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (tab === "fetch") void runFetch();
            else if (tab === "search") void runSearch();
            else void runExtract();
          }}
          className="mt-4 flex gap-2"
        >
          <input
            value={tab === "search" ? query : url}
            onChange={(e) => (tab === "search" ? setQuery(e.target.value) : setUrl(e.target.value))}
            placeholder={tab === "search" ? "Search the web…" : "https://…"}
            className="flex-1 rounded-xl border border-[#e7e0d5] px-3.5 py-2.5 font-mono text-[12.5px] outline-none focus:border-[#ea580c]"
          />
          <button
            type="submit"
            disabled={busy}
            className="rounded-xl px-4 py-2.5 text-[13px] font-medium text-white transition hover:brightness-110 disabled:opacity-40"
            style={{ background: ACCENT }}
          >
            {busy ? "Running…" : tab === "search" ? "Search" : tab === "extract" ? "Extract" : "Fetch"}
          </button>
        </form>

        {tab === "extract" && (
          <p className="mt-2 text-[11px] text-[#a8a29e]">
            Separate multiple URLs with commas. Max 5 per call — billed at 1 credit per 5.
          </p>
        )}

        {error && (
          <div className="mt-4 rounded-xl border border-[#fecaca] bg-[#fef2f2] p-4 text-[12.5px] text-[#991b1b]">
            {error}
          </div>
        )}

        {scrape && (
          <Result
            title={scrape.url}
            backend={scrape.fetcher}
            live={caps?.live ?? false}
            error={scrape.error}
            meta={`${scrape.status || "—"} · ${(scrape.bytes ?? 0).toLocaleString()} bytes`}
          >
            {scrape.markdown && (
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-[#fafaf9] p-3 font-mono text-[11px] leading-relaxed text-[#44403c]">
                {scrape.markdown}
              </pre>
            )}
          </Result>
        )}

        {search && (
          <Result
            title={search.query}
            backend={search.backend}
            live={search.live}
            error={search.error}
          >
            {search.answer && (
              <p className="rounded-lg bg-[#fafaf9] p-3 text-[12.5px] leading-relaxed text-[#44403c]">
                {search.answer}
              </p>
            )}
            <div className="mt-3 space-y-2">
              {search.results.map((h, i) => (
                <motion.a
                  key={h.url + i}
                  href={h.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="block rounded-xl border border-[#f5f5f4] p-3.5 transition hover:border-[#ea580c]/40"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[13px] font-medium">{h.title}</span>
                    <span className="font-mono text-[10.5px] tabular-nums text-[#a8a29e]">
                      {h.score.toFixed(2)}
                    </span>
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[10.5px] text-[#a8a29e]">{h.url}</div>
                  <p className="mt-1.5 text-[12px] leading-relaxed text-[#57534e]">{h.content}</p>
                </motion.a>
              ))}
            </div>
            {search.note && (
              <p className="mt-3 text-[11px] italic leading-relaxed text-[#a8a29e]">{search.note}</p>
            )}
          </Result>
        )}

        {extract && (
          <Result title={`${extract.urls.length} URL(s)`} backend={extract.backend} live={extract.live} error={extract.error}>
            <div className="space-y-3">
              {extract.results.map((r) => (
                <div key={r.url} className="rounded-xl border border-[#f5f5f4] p-3.5">
                  <div className="truncate font-mono text-[10.5px] text-[#a8a29e]">{r.url}</div>
                  <p className="mt-1.5 whitespace-pre-wrap text-[12px] leading-relaxed text-[#57534e]">
                    {r.raw_content}
                  </p>
                </div>
              ))}
            </div>
          </Result>
        )}
      </div>

      <div className="space-y-3">
        {caps && (
          <>
            <Backend
              name="Plain HTTP"
              live={caps.live}
              detail={
                caps.allowlist.length
                  ? `Allowlisted: ${caps.allowlist.join(", ")}`
                  : "No hosts allowlisted, so every live fetch is blocked."
              }
              hint="Free, but cannot render JavaScript. Capped at "
              hintTail={`${caps.max_bytes.toLocaleString()} bytes.`}
            />
            <Backend
              name="Tavily"
              live={caps.tavily.live}
              detail={
                caps.tavily.live
                  ? "Live web search and JS-rendering extraction."
                  : `Set ${caps.tavily.key_env} and ${caps.tavily.enable_env}=1.`
              }
              hint={caps.tavily.free_tier}
            />
          </>
        )}

        <div className="rounded-2xl border border-[#fde68a] bg-[#fffbeb] p-5">
          <div className="text-[12.5px] font-medium text-[#92400e]">
            Both backends gate live calls behind an explicit flag
          </div>
          <p className="mt-1.5 text-[11.5px] leading-relaxed text-[#a16207]">
            A key sitting in a local <code className="font-mono">.env</code> is not enough. Without
            the flag every result is deterministic mock data, labelled as such here and in the API
            payload, so a demo cannot quietly make billable calls or pass placeholders off as
            research.
          </p>
        </div>
      </div>
    </div>
  );
}

function Result({
  title, backend, live, error, meta, children,
}: {
  title: string;
  backend: string;
  live: boolean;
  error?: string;
  meta?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-5">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="rounded-md px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider"
          style={{
            background: live ? "#f0fdf4" : "#fef3c7",
            color: live ? "#166534" : "#92400e",
          }}
        >
          {backend} · {live ? "live" : "mock"}
        </span>
        {meta && <span className="font-mono text-[10.5px] text-[#a8a29e]">{meta}</span>}
      </div>
      <div className="mt-1 truncate font-mono text-[11px] text-[#78716c]">{title}</div>

      {!live && (
        <div className="mt-2 rounded-lg border border-[#fde68a] bg-[#fffbeb] px-3 py-2 text-[11.5px] text-[#92400e]">
          Simulated output — no network call was made. Do not cite or act on these values.
        </div>
      )}

      {error ? (
        <div className="mt-2 rounded-lg border border-[#fecaca] bg-[#fef2f2] p-3 text-[12.5px] text-[#991b1b]">
          {error}
        </div>
      ) : (
        <div className="mt-3">{children}</div>
      )}
    </div>
  );
}

function Backend({
  name, live, detail, hint, hintTail,
}: {
  name: string;
  live: boolean;
  detail: string;
  hint?: string;
  hintTail?: string;
}) {
  return (
    <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-semibold">{name}</span>
        <span
          className="rounded-md px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider"
          style={{
            background: live ? "#f0fdf4" : "#f5f5f4",
            color: live ? "#166534" : "#78716c",
          }}
        >
          {live ? "live" : "mock"}
        </span>
      </div>
      <p className="mt-1.5 text-[11.5px] leading-relaxed text-[#57534e]">{detail}</p>
      {hint && (
        <p className="mt-2 border-t border-[#f5f5f4] pt-2 text-[11px] leading-relaxed text-[#a8a29e]">
          {hint}
          {hintTail}
        </p>
      )}
    </div>
  );
}
