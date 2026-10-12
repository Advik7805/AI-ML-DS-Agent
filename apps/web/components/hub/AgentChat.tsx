"use client";

/**
 * The hub's agent console.
 *
 * The point is not the chat bubble — it is the trace. Every answer shows which bodies the
 * agent reached, what arguments it passed, and how long each step took. That is what turns
 * "an LLM said something" into something you can audit.
 */

import { motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";

import { WakeRetry } from "@/components/body/WakeRetry";
import { clientFetch } from "@/lib/api";

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
  session_context?: string | null;
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
  "Rank the top customers most likely to churn",
  "Which engines should the maintenance crew inspect first?",
  "What patterns make a transaction look suspicious?",
  "Search for the refund window and cite the source",
];

export function AgentChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [provider, setProvider] = useState<string>("connecting");
  const [note, setNote] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [sessionSources, setSessionSources] = useState<string[]>([]);
  const [agentError, setAgentError] = useState<string | null>(null);
  const [agentBusy, setAgentBusy] = useState(true);
  const [progress, setProgress] = useState<TraceEntry[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadAgent = useCallback(async () => {
    setAgentBusy(true);
    setAgentError(null);
    try {
      const info = await clientFetch<{ provider: string; tools: string[]; note: string | null }>("/api/v1/agent");
      setProvider(info.provider);
      setNote(info.note);
    } catch (reason) {
      setProvider("unavailable");
      setAgentError(reason instanceof Error ? reason.message : String(reason));
      setNote(null);
    } finally {
      setAgentBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadAgent();
  }, [loadAgent]);

  useEffect(() => {
    function refreshSession(event?: Event) {
      const detail = event instanceof CustomEvent
        ? event.detail as { sessionId?: unknown; sources?: unknown }
        : undefined;
      try {
        const storedId = window.localStorage.getItem("saros-rag-session") ?? "";
        setSessionId(typeof detail?.sessionId === "string" ? detail.sessionId : storedId);
        const saved = JSON.parse(window.localStorage.getItem("saros-session-sources") ?? "[]") as unknown;
        const sources = Array.isArray(detail?.sources) ? detail.sources : saved;
        setSessionSources(Array.isArray(sources) ? sources.filter((item): item is string => typeof item === "string") : []);
      } catch {
        setSessionId(typeof detail?.sessionId === "string" ? detail.sessionId : "");
        setSessionSources(Array.isArray(detail?.sources)
          ? detail.sources.filter((item): item is string => typeof item === "string")
          : []);
      }
    }
    refreshSession();
    window.addEventListener("saros:session-updated", refreshSession);
    return () => window.removeEventListener("saros:session-updated", refreshSession);
  }, []);

  useEffect(() => {
    function prefill(event: Event) {
      const question = (event as CustomEvent<string>).detail;
      if (typeof question !== "string" || !question.trim()) return;
      setInput(question);
      window.requestAnimationFrame(() => inputRef.current?.focus());
    }
    window.addEventListener("saros:prefill-agent", prefill);
    return () => window.removeEventListener("saros:prefill-agent", prefill);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setBusy(true);
    setProgress([]);
    const history = turns
      .filter((turn) => turn.content)
      .slice(-8)
      .map((turn) => ({ role: turn.role, content: turn.content }));
    setTurns((current) => [...current, { role: "user", content: message }]);

    const trace: TraceEntry[] = [];
    let answer = "";
    let usedProvider = provider;
    try {
      const response = await fetch("/api/v1/agent/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history, ...(sessionId ? { session_id: sessionId } : {}) }),
      });
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const body = (await response.json()) as { detail?: string };
          if (body.detail) detail = body.detail;
        } catch {
          /* keep the HTTP status as the useful error */
        }
        throw new Error(detail);
      }
      if (!response.body) throw new Error("The browser could not open the agent progress stream.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const acceptFrame = (frame: string) => {
        const raw = frame
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!raw) return;
        const update = JSON.parse(raw) as TraceEntry & { answer?: string };
        if (update.node === "done") {
          answer = update.answer ?? "";
          return;
        }
        trace.push(update);
        setProgress((current) => [...current, update]);
        if (update.provider) {
          usedProvider = update.provider;
          setProvider(update.provider);
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\r\n/g, "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          acceptFrame(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf("\n\n");
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) acceptFrame(buffer);

      const toolsUsed = trace.filter((entry) => entry.tool).map((entry) => entry.tool!);
      setTurns((current) => [
        ...current,
        {
          role: "assistant",
          content: answer || "The agent finished without a final answer. Try rephrasing the question.",
          trace,
          tools_used: toolsUsed,
          provider: usedProvider,
        },
      ]);
    } catch (reason) {
      const messageText = reason instanceof Error ? reason.message : String(reason);
      setTurns((current) => [
        ...current,
        {
          role: "assistant",
          content: `Agent unreachable: ${messageText}`,
          trace,
          tools_used: trace.filter((entry) => entry.tool).map((entry) => entry.tool!),
          error: "network",
        },
      ]);
    } finally {
      setBusy(false);
      setProgress([]);
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
          title={`Active language-model provider: ${provider}`}
          className="rounded-md border border-line bg-void-3 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-ink-dim"
          aria-live="polite"
        >
          LLM · {provider}
        </span>
      </div>

      {agentBusy && <p role="status" className="mt-2 text-[11px] text-ink-faint">Waking the API and loading agent status…</p>}
      {note && <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">{note}</p>}
      {agentError && <div className="mt-3"><WakeRetry error={`Agent status unavailable: ${agentError}`} onRetry={() => void loadAgent()} busy={agentBusy} label="Retry status" /></div>}
      <p className="mt-3 rounded-lg border border-line bg-void-3/70 px-3 py-2 text-[10.5px] leading-relaxed text-ink-faint">
        {sessionSources.length > 0
          ? `Active upload context: ${sessionSources.slice(0, 4).join(", ")}${sessionSources.length > 4 ? ` +${sessionSources.length - 4} more` : ""}. The agent can search these temporary session sources and cite matching files.`
          : "No uploads are indexed in this browser session yet. Upload a supported CSV, TXT, Markdown or PDF from Analyze your data to make its analysis or text available to the agent."}
      </p>

      <div className="mt-4 max-h-[420px] min-h-[120px] space-y-3 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <p className="py-6 text-center text-[12.5px] text-ink-faint">
            Ask the agent to reach into any body. It will show its work.
          </p>
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
                        <span className={e.ok === false ? "text-churn" : "text-ink-dim"}>
                          {e.node === "act" ? `↳ ${e.tool}` : `plan${e.provider ? ` (${e.provider})` : ""}`}
                        </span>
                        <span className="ml-2 text-ink-faint">{e.ms}ms</span>
                        {e.error && <div className="mt-0.5 text-churn">{e.error}</div>}
                        {e.arguments && Object.keys(e.arguments).length > 0 && (
                          <div className="mt-0.5 truncate text-ink-faint">
                            {JSON.stringify(e.arguments).slice(0, 120)}
                          </div>
                        )}
                        {e.session_context && <div className="mt-0.5 text-ink-faint">RAG scope: {e.session_context}</div>}
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
          <div role="status" aria-live="polite" className="rounded-xl border border-line bg-void-3 p-3">
            <div className="flex items-center gap-2 text-[11.5px] text-ink-dim">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-hub" />
              {progress.length === 0
                ? "Waking the API and waiting for the agent's first step…"
                : "Agent is working. Progress updates appear as tools finish."}
            </div>
            {progress.length > 0 && (
              <ol className="mt-2 space-y-1 border-l border-line pl-3">
                {progress.map((step, index) => (
                  <li key={`${step.node}-${step.tool ?? "plan"}-${index}`} className="font-mono text-[10px] text-ink-faint">
                    {step.node === "act" ? `↳ ${step.tool} completed` : `plan · ${step.tool_calls?.join(", ") || "compose answer"}`}
                    {typeof step.ms === "number" && <span className="ml-2">{step.ms}ms</span>}
                    {step.error && <span className="ml-2 text-churn">{step.error}</span>}
                  </li>
                ))}
              </ol>
            )}
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
