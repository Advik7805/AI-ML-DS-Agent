"use client";

/**
 * The hub's agent console.
 *
 * The point is not the chat bubble — it is the trace. Every answer shows which bodies the
 * agent reached, what arguments it passed, and how long each step took. That is what turns
 * "an LLM said something" into something you can audit.
 */

import { motion } from "motion/react";
import { useEffect, useRef, useState } from "react";

import { clientFetch } from "@/lib/api";
import { readSessionId, useSessionId } from "@/lib/session";

interface TraceEntry {
  node?: string;
  tool?: string;
  ok?: boolean;
  error?: string | null;
  ms?: number;
  provider?: string;
  tool_calls?: string[];
  arguments?: Record<string, unknown>;
  note?: string;
}

interface Turn {
  role: "user" | "assistant";
  content: string;
  trace?: TraceEntry[];
  tools_used?: string[];
  provider?: string;
  error?: string | null;
}

const SUGGESTIONS = [
  "What can this platform do?",
  "Rank the top 5 customers most likely to churn",
  "How good is the fraud model, and at what operating point?",
  "What is the refund window, and which document says so?",
  "Which engines need maintenance first?",
  "What data is the pricing model trained on?",
];

export function AgentChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState<string>("");
  const [live, setLive] = useState<boolean | null>(null);
  const [note, setNote] = useState<string | null>(null);
  // Counted from the API rather than written into the copy — see the note in the empty state.
  const [toolCount, setToolCount] = useState(0);
  const [bodyCount, setBodyCount] = useState(0);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // The same id the upload studio uses, so a file dropped there is visible here.
  const session = useSessionId();

  useEffect(() => {
    clientFetch<{ provider: string; live: boolean; tools: string[]; note: string | null }>(
      "/api/v1/agent",
    )
      .then((info) => {
        setProvider(info.provider);
        setLive(info.live);
        setNote(info.note);
        setToolCount(info.tools?.length ?? 0);
      })
      .catch(() => setNote("Agent endpoint unreachable."));
  }, []);

  useEffect(() => {
    clientFetch<unknown[]>("/api/v1/modules")
      .then((mods) => setBodyCount(Array.isArray(mods) ? mods.length : 0))
      .catch(() => setBodyCount(0));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  // Guide cards and the upload studio dispatch this. The question is sent, not just typed in:
  // a sample question that fills the box and stops leaves the user one click short, and a click
  // that appears to do nothing is the exact failure this project is trying to avoid. `busyRef`
  // keeps the listener (attached once) from sending while a previous answer is still streaming.
  const busyRef = useRef(busy);
  busyRef.current = busy;
  const sendRef = useRef(send);
  sendRef.current = send;

  useEffect(() => {
    const onAsk = (e: Event) => {
      const detail = (e as CustomEvent<string>).detail;
      if (typeof detail !== "string" || !detail.trim()) return;
      setInput(detail);
      inputRef.current?.focus();
      if (!busyRef.current) void sendRef.current(detail);
    };
    window.addEventListener("saros:ask", onAsk);
    return () => window.removeEventListener("saros:ask", onAsk);
  }, []);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    const history = turns
      .filter((t) => t.content)
      .slice(-8)
      .map((t) => ({ role: t.role, content: t.content }));
    setTurns((t) => [...t, { role: "user", content: message }]);

    try {
      const res = await clientFetch<{
        answer: string;
        provider: string;
        tools_used: string[];
        trace: TraceEntry[];
        error: string | null;
      }>("/api/v1/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // session_id is what makes an uploaded file visible to the agent. Read fresh rather than
        // from render state: the id can be minted by a click that happened after this render.
        body: JSON.stringify({ message, history, session_id: session || readSessionId() }),
      });
      setTurns((t) => [
        ...t,
        {
          role: "assistant",
          content: res.answer,
          trace: res.trace,
          tools_used: res.tools_used,
          provider: res.provider,
          error: res.error,
        },
      ]);
    } catch (e) {
      setTurns((t) => [
        ...t,
        { role: "assistant", content: `Agent unreachable: ${e instanceof Error ? e.message : String(e)}`, error: "network" },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-hub opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-hub" />
          </span>
          <span className="text-[13px] font-semibold">Hub agent</span>
        </div>
        <span
          className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest text-ink-faint"
          title={
            live === false
              ? "No LLM key configured — responses are templated from real tool output."
              : live === true
                ? "A real model is reasoning over the tool results."
                : undefined
          }
        >
          {live !== null && (
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: live ? "#34d399" : "var(--color-ink-faint)" }}
            />
          )}
          {provider ? (
            <>
              {provider}
              <span className={live ? "text-emerald-400/80" : "text-ink-faint"}>
                {live ? "· live" : "· mock"}
              </span>
            </>
          ) : (
            "connecting"
          )}
        </span>
      </div>

      {note && <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">{note}</p>}

      <div className="mt-4 max-h-[420px] min-h-[120px] space-y-3 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="py-6 text-center">
            <p className="text-[12.5px] text-ink-dim">
              Ask the agent to reach into any body. It plans, calls the tools, and shows its work.
            </p>
            <p className="mt-1.5 text-[11.5px] text-ink-faint">
              {/* Counted from the live registry, never typed: the number moved from 22 to 23 and
                  back in this project's own history, and a wrong count is a small lie. */}
              {toolCount > 0 ? `${toolCount} tools` : "Every tool"} across {bodyCount || "every"}{" "}
              bod{bodyCount === 1 ? "y" : "ies"} · upload a file under <em>Your data</em> and the
              agent can answer about it in this conversation.
            </p>
          </div>
        )}

        {turns.map((t, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className={t.role === "user" ? "flex justify-end" : ""}
          >
            <div
              className={
                t.role === "user"
                  ? "max-w-[80%] rounded-2xl rounded-br-sm bg-hub/25 px-3.5 py-2 text-[13px] ring-1 ring-hub/40"
                  : "max-w-[92%] rounded-2xl rounded-bl-sm bg-white/[0.04] px-3.5 py-2.5 text-[13px] ring-1 ring-white/10"
              }
            >
              <div className="whitespace-pre-wrap leading-relaxed text-ink">{renderMarkdown(t.content)}</div>

              {t.trace && t.trace.length > 0 && (
                <details className="mt-2.5">
                  <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-widest text-ink-faint hover:text-ink-dim">
                    trace · {t.tools_used?.length ?? 0} tool{t.tools_used?.length === 1 ? "" : "s"}
                  </summary>
                  <ol className="mt-2 space-y-1.5 border-l border-white/10 pl-3">
                    {t.trace.map((e, j) => (
                      <li key={j} className="font-mono text-[10.5px] leading-relaxed">
                        <span className={e.ok === false ? "text-red-400" : "text-ink-dim"}>
                          {e.node === "act" ? `↳ ${e.tool}` : `plan${e.provider ? ` (${e.provider})` : ""}`}
                        </span>
                        <span className="ml-2 text-ink-faint">{e.ms}ms</span>
                        {e.error && <div className="mt-0.5 text-red-400">{e.error}</div>}
                        {e.arguments && Object.keys(e.arguments).length > 0 && (
                          <div className="mt-0.5 truncate text-ink-faint">
                            {JSON.stringify(e.arguments).slice(0, 120)}
                          </div>
                        )}
                        {e.note && <div className="mt-0.5 text-churn">{e.note}</div>}
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </div>
          </motion.div>
        ))}

        {busy && (
          <div className="flex items-center gap-2 text-[12px] text-ink-faint">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-hub" />
            planning…
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
        className="mt-4 flex gap-2"
      >
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask across the bodies…"
          className="flex-1 rounded-xl border border-white/10 bg-white/[0.04] px-3.5 py-2.5 text-[13px] text-ink outline-none placeholder:text-ink-faint focus:border-hub/60"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-xl bg-hub px-4 py-2.5 text-[13px] font-medium text-white transition hover:brightness-110 disabled:opacity-40"
        >
          Send
        </button>
      </form>

      {turns.length === 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => void send(s)}
              className="rounded-full border border-white/10 px-2.5 py-1 text-[11px] text-ink-dim transition hover:border-hub/50 hover:text-ink"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Minimal markdown: bold, code, and line breaks. No dependency, no XSS surface. */
function renderMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**"))
      return (
        <strong key={i} className="font-semibold text-white">
          {p.slice(2, -2)}
        </strong>
      );
    if (p.startsWith("`") && p.endsWith("`"))
      return (
        <code key={i} className="rounded bg-white/10 px-1 font-mono text-[11.5px]">
          {p.slice(1, -1)}
        </code>
      );
    return <span key={i}>{p}</span>;
  });
}
