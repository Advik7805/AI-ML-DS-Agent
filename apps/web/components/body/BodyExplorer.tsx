"use client";

/**
 * Generic body explorer — renders *any* Nexus body from its manifest and tool schemas.
 *
 * This is the payoff of the module contract on the frontend: one component drives churn, fraud,
 * maintenance, pricing, diagnostics, RAG and the scraper. A new body gets a working console
 * without a line of bespoke UI, because everything here is derived from the JSON Schema the
 * body already publishes for the agent.
 */

import { motion } from "motion/react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { clientFetch } from "@/lib/api";
import type { AgentTool, Explanation, MetricsReport, ModuleManifest, Prediction } from "@/lib/types";

interface Props {
  moduleId: string;
  accent: string;
}

type Json = Record<string, any>;

function riskColor(score: number, accent: string) {
  if (score >= 0.75) return "#dc2626";
  if (score >= 0.5) return "#ea580c";
  if (score >= 0.25) return "#d97706";
  return accent;
}

function sampleValue(schema: Json): any {
  const types: string[] = Array.isArray(schema?.type) ? schema.type : [schema?.type];
  if ("default" in (schema ?? {})) return schema.default;
  if ("examples" in (schema ?? {})) return schema.examples[0];
  if (types.includes("number") || types.includes("integer")) return 0;
  if (types.includes("string")) return "";
  return null;
}

export function BodyExplorer({ moduleId, accent }: Props) {
  const [manifest, setManifest] = useState<ModuleManifest | null>(null);
  const [metrics, setMetrics] = useState<MetricsReport | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [toolName, setToolName] = useState<string>("");
  const [args, setArgs] = useState<Json>({});
  const [result, setResult] = useState<Json | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [raw, setRaw] = useState<string>("");

  useEffect(() => {
    clientFetch<ModuleManifest>(`/api/v1/modules/${moduleId}`)
      .then((m) => {
        setManifest(m);
        // Default to the first tool that takes structured input, so the console opens useful.
        const primary =
          m.tools.find((t) => Object.keys(t.parameters_schema?.properties ?? {}).length > 3) ?? m.tools[0];
        if (primary) {
          setToolName(primary.name);
          setArgs(defaultArgs(primary));
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));

    clientFetch<MetricsReport>(`/api/v1/modules/${moduleId}/metrics`)
      .then(setMetrics)
      .catch((e) => setMetricsError(e instanceof Error ? e.message : String(e)));
  }, [moduleId]);

  const tool = useMemo(
    () => manifest?.tools.find((t) => t.name === toolName) ?? null,
    [manifest, toolName],
  );

  function defaultArgs(t: AgentTool): Json {
    const props = (t.parameters_schema?.properties ?? {}) as Json;
    const out: Json = {};
    for (const [k, v] of Object.entries(props)) out[k] = sampleValue(v as Json);
    return out;
  }

  const run = useCallback(async () => {
    if (!tool) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setPrediction(null);
    setExplanation(null);
    try {
      const res = await clientFetch<Json>(`/api/v1/agent/tools/${tool.name}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arguments: coerce(args) }),
      });
      setResult(res.result as Json);
      setRaw(JSON.stringify(res.result, null, 2));
      // If the tool returns a Prediction shape, render it richly.
      const r = res.result as Json;
      if (r && typeof r.score === "number" && typeof r.label === "string") {
        setPrediction(r as unknown as Prediction);
      }
      const feats = predictionFeatures(r);
      if (feats) {
        try {
          setExplanation(await clientFetch<Explanation>(`/api/v1/modules/${moduleId}/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: feats }),
          }));
        } catch {
          /* not every body explains */
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [tool, args, moduleId]);

  const props = (tool?.parameters_schema?.properties ?? {}) as Json;
  const required = new Set((tool?.parameters_schema?.required ?? []) as string[]);

  return (
    <div className="mx-auto max-w-6xl px-5 pb-16">
      {/* --- summary + metrics ------------------------------------------- */}
      {manifest && (
        <section id="model" className="scroll-mt-24 pb-8">
          <p className="max-w-3xl text-[15px] leading-relaxed text-[#57534e]">{manifest.summary}</p>

          <div className="mt-5 flex flex-wrap gap-2">
            {manifest.capabilities.map((c) => (
              <span
                key={c}
                className="rounded-full border px-2.5 py-1 font-mono text-[10.5px] uppercase tracking-wider"
                style={{ borderColor: `${accent}55`, color: accent }}
              >
                {c}
              </span>
            ))}
          </div>

          {metrics && (
            <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {Object.entries(metrics.scores)
                .slice(0, 8)
                .map(([k, v], i) => (
                  <div
                    key={k}
                    className={`rounded-2xl border p-4 shadow-sm ${
                      i === 0 ? "bg-white" : "border-[#e7e0d5] bg-white"
                    }`}
                    style={i === 0 ? { borderColor: `${accent}88` } : undefined}
                  >
                    <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#a8a29e]">
                      {k.replace(/_/g, " ")}
                    </div>
                    <div className="mt-1.5 text-2xl font-semibold tabular-nums tracking-tight">
                      {typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)) : "—"}
                    </div>
                  </div>
                ))}
            </div>
          )}

          {metrics && (metrics.groups ? Object.keys(metrics.groups).length > 0 : false) && (
            <div className="mt-3 rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
              <div className="text-[13px] font-semibold">Group breakdown</div>
              <div className="mt-1 font-mono text-[10.5px] text-[#a8a29e]">{metrics.group_basis}</div>
              <table className="mt-3 w-full text-left text-[12px]">
                <tbody>
                  {Object.entries(metrics.groups).map(([g, vals]) => (
                    <tr key={g} className="border-b border-[#f5f5f4] last:border-0">
                      <td className="py-1.5 font-mono text-[11.5px]">{g}</td>
                      {Object.entries(vals).map(([k, v]) => (
                        <td key={k} className="py-1.5 text-[11.5px] tabular-nums text-[#57534e]">
                          {k.replace(/_/g, " ")} {(v * 100).toFixed(1)}%
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {metrics && (
            <details className="mt-3 rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
              <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.14em] text-[#a8a29e] hover:text-[#57534e]">
                Methodology notes ({metrics.notes.length})
              </summary>
              <ul className="mt-3 space-y-2 border-t border-[#f5f5f4] pt-3">
                {metrics.notes.map((n) => (
                  <li key={n} className="text-[12px] leading-relaxed text-[#57534e]">
                    <span className="mr-2" style={{ color: accent }}>—</span>
                    {n}
                  </li>
                ))}
              </ul>
            </details>
          )}

          {metricsError && (
            <div className="mt-3 rounded-xl border border-[#fde68a] bg-[#fffbeb] p-4 text-[12.5px] text-[#78350f]">
              Model card unavailable: {metricsError}. The console below still works.
            </div>
          )}
        </section>
      )}

      {/* --- console ------------------------------------------------------ */}
      <section id="console" className="scroll-mt-24">
        <div className="grid gap-5 lg:grid-cols-[380px_minmax(0,1fr)]">
          <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">Tool</div>
            <select
              value={toolName}
              onChange={(e) => {
                const t = manifest?.tools.find((x) => x.name === e.target.value);
                setToolName(e.target.value);
                if (t) setArgs(defaultArgs(t));
              }}
              className="mt-1.5 w-full rounded-md border border-[#e7e0d5] bg-white px-2.5 py-1.5 font-mono text-[12px]"
            >
              {(manifest?.tools ?? []).map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name}
                </option>
              ))}
            </select>

            {tool && <p className="mt-2.5 text-[11.5px] leading-relaxed text-[#78716c]">{tool.description}</p>}

            <div className="mt-4 max-h-[46vh] space-y-2.5 overflow-y-auto pr-1">
              {Object.entries(props).map(([k, v]) => {
                const sch = v as Json;
                const types: string[] = Array.isArray(sch.type) ? sch.type : [sch.type];
                const numeric = types.includes("number") || types.includes("integer");
                return (
                  <div key={k} className="flex items-center gap-3">
                    <label className="w-40 shrink-0 truncate font-mono text-[10.5px] text-[#57534e]">
                      {k}
                      {required.has(k) && <span style={{ color: accent }}> *</span>}
                    </label>
                    <input
                      value={String(args[k] ?? "")}
                      onChange={(e) =>
                        setArgs((a) => ({ ...a, [k]: numeric && e.target.value !== "" ? Number(e.target.value) : e.target.value }))
                      }
                      className="flex-1 rounded-md border border-[#e7e0d5] px-2 py-1 font-mono text-[11.5px]"
                    />
                  </div>
                );
              })}
            </div>

            <button
              onClick={run}
              disabled={busy}
              className="mt-4 w-full rounded-lg py-2 text-[13px] font-medium text-white transition disabled:opacity-50"
              style={{ background: accent }}
            >
              {busy ? "Running…" : "Run tool"}
            </button>
            <p className="mt-2 text-center font-mono text-[10px] text-[#a8a29e]">
              POST /api/v1/agent/tools/{toolName}
            </p>
          </div>

          <div className="space-y-4">
            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-[12.5px] text-red-700">
                {error}
              </div>
            )}

            {prediction && (
              <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
                <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">Result</div>
                <div className="mt-2 flex items-end gap-3">
                  <span
                    className="text-4xl font-semibold tabular-nums tracking-tight"
                    style={{ color: riskColor(prediction.score, accent) }}
                  >
                    {(prediction.score * 100).toFixed(1)}%
                  </span>
                  <span className="pb-1 text-[13px] font-medium">{prediction.label}</span>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#f5f5f4]">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: riskColor(prediction.score, accent) }}
                    animate={{ width: `${Math.max(2, prediction.score * 100)}%` }}
                    transition={{ type: "spring", stiffness: 140, damping: 20 }}
                  />
                </div>
                <div className="mt-4 rounded-xl bg-[#fafaf9] p-3.5 ring-1 ring-[#e7e0d5]">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#a8a29e]">
                    Recommended
                  </div>
                  <p className="mt-1 text-[12.5px] leading-relaxed text-[#44403c]">{prediction.decision}</p>
                </div>
              </div>
            )}

            {explanation && explanation.attributions.length > 0 && (
              <div className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
                <div className="flex items-baseline justify-between">
                  <div className="text-[14px] font-semibold">Why</div>
                  <span className="font-mono text-[10px] uppercase tracking-widest text-[#a8a29e]">
                    {explanation.kind} · {explanation.space}
                  </span>
                </div>
                <p className="mt-1.5 text-[12px] text-[#57534e]">{explanation.summary}</p>
                <div className="mt-4 space-y-2">
                  {explanation.attributions.slice(0, 8).map((a) => {
                    const max = Math.max(...explanation.attributions.map((x) => Math.abs(x.attribution)), 1e-6);
                    const pct = (Math.abs(a.attribution) / max) * 100;
                    const up = a.direction === "increases_risk";
                    return (
                      <div key={a.feature} className="flex items-center gap-2.5">
                        <span className="w-36 shrink-0 truncate font-mono text-[10px] text-[#57534e]">
                          {a.feature}
                        </span>
                        <div className="relative h-3.5 flex-1 rounded bg-[#f5f5f4]">
                          <div className="absolute left-1/2 top-0 h-full w-px bg-[#e7e0d5]" />
                          <div
                            className="absolute top-0.5 h-2.5 rounded-sm"
                            style={{
                              background: up ? "#dc2626" : "#16a34a",
                              width: `${pct / 2}%`,
                              left: up ? "50%" : `${50 - pct / 2}%`,
                            }}
                          />
                        </div>
                        <span className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums text-[#78716c]">
                          {a.attribution > 0 ? "+" : ""}
                          {a.attribution.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <ul className="mt-4 space-y-1.5 border-t border-[#f5f5f4] pt-3">
                  {explanation.caveats.map((c) => (
                    <li key={c} className="text-[10.5px] leading-relaxed text-[#a8a29e]">— {c}</li>
                  ))}
                </ul>
              </div>
            )}

            {raw && (
              <details className="rounded-2xl border border-[#e7e0d5] bg-white p-5 shadow-sm">
                <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.14em] text-[#a8a29e]">
                  Raw response
                </summary>
                <pre className="mt-3 max-h-96 overflow-auto rounded-lg bg-[#fafaf9] p-3 font-mono text-[10.5px] leading-relaxed text-[#44403c]">
                  {raw}
                </pre>
              </details>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

/** Coerce the form's string values to the types the schema declares. */
function coerce(args: Json): Json {
  const out: Json = {};
  for (const [k, v] of Object.entries(args)) {
    if (v === "" || v === null) out[k] = null;
    else out[k] = v;
  }
  return out;
}

/** If the tool result echoes its inputs, reuse them for the explain call. */
function predictionFeatures(result: any): Json | null {
  if (result && typeof result === "object" && result.inputs && Object.keys(result.inputs).length > 0) {
    return result.inputs as Json;
  }
  return null;
}
