import type { Metadata, Viewport } from "next";
import Link from "next/link";

import "../globals.css";

import { WakeIndicator } from "@/components/body/WakingNotice";

/*
  The HUB root layout.

  There is deliberately no top-level `app/layout.tsx`. Each route group owns its own root
  layout, which is what gives every body its own <html>/<body>, its own font stack, its own
  chrome and its own identity. Next.js documents the trade-off explicitly: navigating between
  different root layouts triggers a full page load — which is exactly the "you have arrived at
  a different site" feeling we want, the same way a retailer's electronics vertical is its own
  storefront rather than a tab.
*/

export const metadata: Metadata = {
  title: "Saros — AI/ML/Data-Science agent platform",
  description:
    "One hub, many independent bodies. A multimodal agent platform assembling the senior data scientist's toolchain: RAG, fraud detection, predictive maintenance, explainable AI, dynamic pricing and churn.",
};

export const viewport: Viewport = {
  themeColor: "#05060a",
  width: "device-width",
  initialScale: 1,
};

export default function HubLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen overflow-x-hidden bg-void">
        <div className="ambient" />
        <div className="pointer-events-none fixed inset-0 z-0 grid-floor" />

        <header className="relative z-20 flex items-center justify-between px-6 py-5 sm:px-10">
          <Link href="/" className="group flex items-center gap-3">
            <span className="relative grid h-8 w-8 place-items-center">
              <span className="absolute inset-0 rotate-45 rounded-[7px] bg-hub/25 ring-1 ring-hub/50 transition-transform duration-500 group-hover:rotate-[135deg]" />
              <span className="relative h-2 w-2 rounded-full bg-hub shadow-[0_0_12px_3px_rgba(124,92,255,0.7)]" />
            </span>
            <span className="text-[15px] font-semibold tracking-tight">
              Saros
              <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">
                hub
              </span>
            </span>
          </Link>

          <nav className="flex items-center gap-1 text-[13px]">
            <Link
              href="#tenant"
              className="rounded-lg px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink"
            >
              The company
            </Link>
            <Link
              href="#upload"
              className="rounded-lg px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink"
            >
              Your data
            </Link>
            <Link
              href="#guide"
              className="rounded-lg px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink"
            >
              Guide
            </Link>
            <Link
              href="#bodies"
              className="rounded-lg px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink"
            >
              Bodies
            </Link>
            <Link
              href="#agent"
              className="rounded-lg px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink"
            >
              Agent
            </Link>
            <Link
              href="#runs"
              className="rounded-lg px-3 py-1.5 text-ink-dim transition hover:bg-white/5 hover:text-ink"
            >
              Runs
            </Link>
            <Link
              href="/churn"
              className="ml-2 rounded-lg bg-white/10 px-3.5 py-1.5 font-medium text-ink ring-1 ring-white/15 transition hover:bg-white/15"
            >
              Enter a body
            </Link>
          </nav>
        </header>

        <main className="relative z-10">{children}</main>

        <WakeIndicator />

        <footer className="relative z-10 border-t border-line px-6 py-10 text-center text-xs text-ink-faint sm:px-10">
          <p>
            Saros — one contract, many bodies. Every module is independently deployable and
            connected to the hub through{" "}
            <code className="font-mono text-ink-dim">core/contract.py</code>.
          </p>
        </footer>
      </body>
    </html>
  );
}
