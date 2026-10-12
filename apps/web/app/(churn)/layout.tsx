import type { Metadata, Viewport } from "next";
import Link from "next/link";

import "../globals.css";

export const metadata: Metadata = {
  title: "Churn & Retention — Saros",
  description:
    "Gradient-boosted churn scoring on the telecom dataset with calibrated risk scores, operating thresholds, and per-customer SHAP explanations.",
};

export const viewport: Viewport = {
  themeColor: "#05060a",
  width: "device-width",
  initialScale: 1,
};

export default function ChurnLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-void text-ink antialiased">
        <div
          className="pointer-events-none fixed inset-0 -z-10"
          style={{
            background:
              "radial-gradient(60rem 30rem at 100% -10%, rgba(245,158,11,0.12), transparent 60%), radial-gradient(40rem 25rem at -10% 20%, rgba(217,119,6,0.06), transparent 60%)",
          }}
        />

        <header className="sticky top-0 z-30 border-b border-line bg-void/85 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3.5">
            <div className="flex items-center gap-3">
              <span className="grid h-7 w-7 place-items-center rounded-md bg-churn text-[13px] font-bold text-white">
                C
              </span>
              <div className="leading-tight">
                <div className="text-[14px] font-semibold tracking-tight">Churn &amp; Retention</div>
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
                  saros body · v0.1.0
                </div>
              </div>
            </div>

            <nav className="flex items-center gap-1 text-[13px]">
              <a href="#scorer" className="rounded-md px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink">
                Score
              </a>
              <a href="#portfolio" className="rounded-md px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink">
                Portfolio
              </a>
              <a href="#model" className="rounded-md px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink">
                Model card
              </a>
              <Link
                href="/"
                className="ml-2 rounded-md border border-line bg-void-2 px-3 py-1.5 font-medium text-ink transition hover:bg-white/5"
              >
                ← Hub
              </Link>
            </nav>
          </div>
        </header>

        <main className="relative z-10">{children}</main>

        <footer className="relative z-10 mt-16 border-t border-line px-5 py-8">
          <div className="mx-auto max-w-6xl text-[12px] text-ink-faint">
            <p>
              An independent Saros body. Own API router, own model, own tests. Connected to the
              hub only through the shared module contract.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
