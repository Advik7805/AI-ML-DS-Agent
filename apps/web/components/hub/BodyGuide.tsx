"use client";

import type { ModuleManifest } from "@/lib/types";

const BODY_LINKS: Record<string, { title: string; href: string; accent: string }> = {
  churn: { title: "Churn & Retention", href: "/churn", accent: "#f59e0b" },
  fraud: { title: "Fraud & Anomaly", href: "/fraud", accent: "#f43f5e" },
  maintenance: { title: "Predictive Maintenance", href: "/maintenance", accent: "#22d3ee" },
  pricing: { title: "Dynamic Pricing", href: "/pricing", accent: "#34d399" },
  rag: { title: "Knowledge RAG", href: "/rag", accent: "#38bdf8" },
  scraper: { title: "Web Scraper", href: "/scraper", accent: "#f97316" },
  xai_health: { title: "Explainable Diagnostics", href: "/diagnostics", accent: "#a78bfa" },
};

const GUIDES = [
  {
    id: "churn",
    purpose: "Find the customers who may leave next and give the retention team a reasoned place to start.",
    accepted: "CSV · telecom customer features; optional churn outcome column",
    questions: [
      "Which five customers should our retention team contact first?",
      "What raises churn risk for a customer with several service calls?",
      "What does the churn model say about a customer on an international plan?",
    ],
  },
  {
    id: "fraud",
    purpose: "Score uploaded transactions, surface unusual patterns and give reviewers a reasoned hold decision.",
    accepted: "CSV · 10 transaction features; optional is_fraud outcome column. Built-in examples are synthetic, not company data.",
    questions: [
      "What patterns make a card transaction look unusual?",
      "How does the fraud model handle a large late-night purchase abroad?",
      "What should an analyst do with a transaction at high fraud risk?",
    ],
  },
  {
    id: "maintenance",
    purpose: "Prioritize the engines most likely to need attention, with remaining-life estimates to plan work.",
    accepted: "CSV or whitespace TXT · turbofan unit, cycle, settings and sensors",
    questions: [
      "Which engines should the maintenance crew inspect first?",
      "How is remaining useful life estimated from turbofan sensors?",
      "What does the model say about an engine showing sensor degradation?",
    ],
  },
  {
    id: "pricing",
    purpose: "Use observed price and demand to estimate SKU elasticity and find bounded price opportunities.",
    accepted: "CSV · sku, price, demand; at least 10 usable rows per SKU",
    questions: [
      "Which products have the clearest opportunity for a price increase?",
      "How does the demand floor constrain a SKU's recommended price?",
      "What data is needed to estimate price elasticity for a product?",
    ],
  },
  {
    id: "rag",
    purpose: "Search built-in policy and your uploaded files, then answer from ranked passages with exact file and heading citations.",
    accepted: "Markdown, TXT or PDF · up to 2 MB extracted text per document; uploads stay in a temporary browser session",
    questions: [
      "Search the knowledge base for the refund window and cite the source.",
      "What discount can a retention agent approve?",
      "What is the service uptime commitment?",
    ],
  },
  {
    id: "scraper",
    purpose: "Turn an allowed public page into structured source material for research and analysis.",
    accepted: "A URL in the scraper console · no file upload",
    questions: [
      "Can you fetch and summarize a public product page?",
      "What safeguards keep the scraper away from private network addresses?",
      "How can scraped text support a knowledge-base answer?",
    ],
  },
  {
    id: "xai_health",
    purpose: "Review a diabetes screening score, understand its leading factors and inspect fairness metrics.",
    accepted: "One patient at a time in the console · no file upload",
    questions: [
      "What factors drive a diabetes risk score?",
      "How is the diabetes model checked for fairness across groups?",
      "Why does this screening model focus on missed positive cases?",
    ],
  },
] as const;

export function BodyGuide({ modules }: { modules: ModuleManifest[] }) {
  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-8 max-w-2xl">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-hub">Choose a starting point</p>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight sm:text-3xl">Guide to the bodies</h2>
        <p className="mt-2 text-[14px] leading-relaxed text-ink-dim">
          Each body is built for a different decision. Pick a sample question to load it into the agent, or open the body for its data console.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {GUIDES.map((guide) => {
          const body = modules.find((module) => module.id === guide.id);
          const known = BODY_LINKS[guide.id];
          const status = body?.status ?? "live";
          return (
            <article key={guide.id} className="glass flex h-full flex-col rounded-2xl p-5">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-[15px] font-semibold" style={{ color: known.accent }}>
                  {body?.title ?? known.title}
                </h3>
                <span className="font-mono text-[9px] uppercase tracking-widest text-ink-faint">{status}</span>
              </div>
              <p className="mt-2 min-h-[3.5rem] text-[12.5px] leading-relaxed text-ink-dim">{guide.purpose}</p>
              <p className="mt-3 rounded-lg border border-line bg-void-3/70 px-3 py-2 text-[10.5px] leading-relaxed text-ink-faint">
                <span className="font-mono uppercase tracking-wider">Input · </span>{guide.accepted}
              </p>
              <div className="mt-4 flex-1">
                <p className="font-mono text-[9px] uppercase tracking-[0.15em] text-ink-faint">Try asking</p>
                <ul className="mt-2 space-y-1.5">
                  {guide.questions.map((question) => (
                    <li key={question}>
                      <button
                        type="button"
                        onClick={() => {
                          window.dispatchEvent(new CustomEvent("saros:prefill-agent", { detail: question }));
                          document.getElementById("agent")?.scrollIntoView({ behavior: "smooth", block: "start" });
                        }}
                        className="w-full rounded-lg border border-line/70 px-3 py-2 text-left text-[11px] leading-relaxed text-ink-dim transition hover:border-hub/50 hover:bg-void-3 hover:text-ink"
                      >
                        {question}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
              <a
                href={body?.routes?.ui ?? known.href}
                className="mt-4 inline-flex items-center gap-1 text-[11px] font-medium text-ink-faint transition hover:text-ink"
              >
                Open body <span aria-hidden="true">→</span>
              </a>
            </article>
          );
        })}
      </div>
    </div>
  );
}
