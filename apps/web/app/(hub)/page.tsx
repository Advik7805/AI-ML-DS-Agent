import Link from "next/link";

import { AgentChat } from "@/components/hub/AgentChat";
import { IntegrationsPanel } from "@/components/hub/IntegrationsPanel";
import { RunLog } from "@/components/hub/RunLog";
import { ModuleGraph } from "@/components/hub/ModuleGraph";
import { TenantPanel } from "@/components/hub/TenantPanel";
import { getModules } from "@/lib/api";
import type { ModuleManifest } from "@/lib/types";
import { KNOWN_BODIES } from "@/lib/types";

export const dynamic = "force-dynamic";

async function loadModules(): Promise<{ modules: ModuleManifest[]; error: string | null }> {
  try {
    return { modules: await getModules(true), error: null };
  } catch (err) {
    // The hub must still render when the API is cold — a free-tier backend sleeps, and a blank
    // page is the worst possible way to communicate "waking up".
    return { modules: [], error: err instanceof Error ? err.message : String(err) };
  }
}

export default async function HubPage() {
  const { modules, error } = await loadModules();
  const live = new Map(modules.map((m) => [m.id, m]));

  // Every body the API reports, rendered from its live manifest; when the backend is asleep the
  // static mirror keeps the grid clickable. The API's statuses win over the mirror.
  const tiles: BodyTileProps[] = modules.length
    ? modules.map((m) => ({
        id: m.id,
        title: m.title,
        blurb: m.summary,
        accent: m.accent,
        status: m.status,
        tools: m.tools.map((t) => t.name),
        href: m.routes?.ui,
      }))
    : KNOWN_BODIES.map((b) => ({
        id: b.id,
        title: b.title,
        blurb: b.blurb,
        accent: b.accent,
        status: "live" as const,
        tools: [],
        href: b.href,
      }));
  const liveCount = tiles.filter((t) => t.status !== "planned").length;
  const plannedCount = tiles.length - liveCount;

  return (
    <>
      {/* --- hero: the 3D graph IS the navigation ----------------------------- */}
      <section className="px-6 pt-6 sm:px-10">
        <div className="mx-auto max-w-3xl text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-hub">
            {liveCount} live · {plannedCount} planned
          </p>
          <h1 className="mt-5 text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
            One hub.
            <br />
            <span className="bg-gradient-to-r from-hub via-sky-400 to-churn bg-clip-text text-transparent">
              Many independent bodies.
            </span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-[15px] leading-relaxed text-ink-dim">
            Ask in plain English, or drop in your own CSV, PDF or text file — Saros scores churn,
            flags fraud transaction by transaction, forecasts which engine fails next, prices SKUs
            and answers policy questions, then explains every decision with cited evidence. All
            seven tools run on one fictional company&rsquo;s data, so the numbers connect instead of
            floating free.
          </p>
        </div>

        <ModuleGraph live={modules} />

        <div id="tenant" className="mx-auto mt-10 max-w-6xl scroll-mt-24">
          <TenantPanel modules={modules} />
        </div>

        {error && (
          <div className="mx-auto mt-2 max-w-xl rounded-xl border border-churn/30 bg-churn/5 px-4 py-3 text-center text-xs text-churn">
            Backend not reachable yet ({error}). Free tiers sleep — the hub still renders, and
            each body retries on its own.
          </div>
        )}
      </section>

      {/* --- bodies ---------------------------------------------------------- */}
      <section id="bodies" className="scroll-mt-20 px-6 py-20 sm:px-10">
        <div className="mx-auto max-w-6xl">
          <div className="mb-10 flex items-end justify-between">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight">The bodies</h2>
              <p className="mt-2 text-sm text-ink-dim">
                Each is its own storefront — own route, own identity, own models, own tests.
              </p>
            </div>
            <span className="hidden font-mono text-[11px] uppercase tracking-widest text-ink-faint sm:block">
              core/contract.py
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {tiles.map((t) => (
              <BodyTile key={t.id} {...t} />
            ))}
          </div>
        </div>
      </section>

      {/* --- agent ----------------------------------------------------------- */}
      <section id="agent" className="scroll-mt-20 border-t border-line px-6 py-20 sm:px-10">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-semibold tracking-tight">The connective tissue</h2>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-dim">
            The hub is not a landing page with links. Every body registers its capabilities
            into one tool registry, so the agent can plan across them — and because every body
            returns the same{" "}
            <code className="font-mono text-ink">Prediction</code> and{" "}
            <code className="font-mono text-ink">Explanation</code> shapes, one component
            renders &ldquo;why?&rdquo; for all of them.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            <Stat label="Tools registered" value={String(modules.reduce((n, m) => n + m.tools.length, 0))} />
            <Stat label="Bodies loaded" value={String(modules.length)} />
            <Stat label="Shared contract" value="1" />
          </div>

          <div className="mt-8">
            <AgentChat />
          </div>
        </div>
      </section>

      {/* --- integrations ------------------------------------------------------ */}
      <section id="integrations" className="scroll-mt-20 border-t border-line px-6 py-20 sm:px-10">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-semibold tracking-tight">Runs with nothing, lights up with keys</h2>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-dim">
            No environment variable is required. Every external service degrades to a
            deterministic mock that labels itself, so the platform is a working product on an
            empty <code className="font-mono text-ink">.env</code> and gains real capabilities one
            key at a time. A key alone never enables a metered service — each also needs its
            explicit flag, so a local experiment cannot make a demo billable.
          </p>
          <div className="mt-8">
            <IntegrationsPanel />
          </div>
        </div>
      </section>

      <section id="runs" className="scroll-mt-20 border-t border-line px-6 py-20 sm:px-10">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-semibold tracking-tight">Every training run is recorded</h2>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-dim">
            Seven bodies, each selecting an estimator by a metric, makes &ldquo;which run produced
            the model that is loaded&rdquo; unanswerable without a log. Training is hooked at the
            contract level, so a body added later cannot forget to record itself. Runs go to MLflow
            when <code className="font-mono text-ink">MLFLOW_TRACKING_URI</code> is set and to a
            local JSONL file otherwise — the panel says which, and says plainly that a local log
            on an ephemeral filesystem is lost on redeploy.
          </p>
          <div className="mt-8">
            <RunLog />
          </div>
        </div>
      </section>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-xl px-5 py-4">
      <div className="text-2xl font-semibold tabular-nums">{value}</div>
      <div className="mt-1 text-[11px] uppercase tracking-wider text-ink-faint">{label}</div>
    </div>
  );
}

interface BodyTileProps {
  id: string;
  title: string;
  blurb: string;
  accent: string;
  status: "live" | "beta" | "planned";
  tools: string[];
  href?: string;
}

function BodyTile({ id, title, blurb, accent, status, tools, href }: BodyTileProps) {
  // "beta" is enterable too — it is a running body, just labelled as not yet stable.
  const enterable = status !== "planned" && Boolean(href);
  const inner = (
    <div
      className="glass group relative h-full overflow-hidden rounded-2xl p-5 transition duration-300 hover:-translate-y-1"
      style={{ borderColor: enterable ? `${accent}55` : undefined }}
    >
      <div
        className="absolute -right-10 -top-10 h-28 w-28 rounded-full opacity-20 blur-2xl transition-opacity duration-500 group-hover:opacity-45"
        style={{ background: accent }}
      />
      <div className="relative">
        <div className="flex items-center justify-between">
          <span className="h-2.5 w-2.5 rotate-45 rounded-[2px]" style={{ background: accent }} />
          <span
            className="font-mono text-[10px] uppercase tracking-widest"
            style={{ color: status !== "planned" ? accent : "#5b6382" }}
          >
            {status}
          </span>
        </div>

        <h3 className="mt-4 text-[15px] font-semibold tracking-tight">{title}</h3>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-dim">{blurb}</p>

        {tools.length > 0 && (
          <ul className="mt-4 space-y-1">
            {tools.map((t) => (
              <li key={t} className="font-mono text-[11px] text-ink-faint">
                · {t}
              </li>
            ))}
          </ul>
        )}

        <div className="mt-5 flex items-center justify-between">
          <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">/{id}</span>
          {enterable && (
            <span className="text-[12px] font-medium" style={{ color: accent }}>
              enter →
            </span>
          )}
        </div>
      </div>
    </div>
  );

  return enterable && href ? (
    <Link href={href} className="block h-full">
      {inner}
    </Link>
  ) : (
    <div className="h-full opacity-70">{inner}</div>
  );
}
