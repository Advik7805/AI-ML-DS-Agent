import type { Metadata, Viewport } from "next";
import Link from "next/link";

import "@/app/globals.css";

import { BackdropVideo } from "./BackdropVideo";
import { WakeIndicator } from "./WakingNotice";
import { StorefrontField } from "./StorefrontField";

/**
 * Shared storefront chrome, parameterised by body.
 *
 * Each body still gets its own `layout.tsx` — that is what makes it a separate root layout and
 * therefore a separate document with its own `<html>`, theme-colour and metadata. This file
 * only supplies the parts that are genuinely common, so a body can diverge freely without
 * duplicating two hundred lines of nav.
 */
export interface StorefrontProps {
  id: string;
  title: string;
  tagline: string;
  accent: string;
  sections: Array<{ href: string; label: string }>;
}

export function storefrontMetadata(p: StorefrontProps): Metadata {
  return {
    title: `${p.title} — Saros`,
    description: `${p.tagline} An independent Saros body connected to the hub through the shared module contract.`,
  };
}

/**
 * Next 14+ moved `themeColor` out of `metadata` into its own `viewport` export, so each root
 * layout sets it separately. This is what makes the browser chrome change colour per body —
 * the cheapest possible signal that you have left one site for another.
 */
export function storefrontViewport(p: StorefrontProps): Viewport {
  return { themeColor: p.accent };
}

export function Storefront({ p, children }: { p: StorefrontProps; children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-void text-ink antialiased">
        {/* Video sits under the field: the procedural points stay legible over footage, and
            with no BACKDROP_VIDEO_URL set the video renders nothing at all. */}
        <BackdropVideo accent={p.accent} />
        <StorefrontField accent={p.accent} />

        <header className="sticky top-0 z-30 border-b border-line bg-void/85 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3.5">
            <div className="flex items-center gap-3">
              <span
                className="grid h-7 w-7 place-items-center rounded-md text-[13px] font-bold text-white"
                style={{ background: p.accent }}
              >
                {p.title[0]}
              </span>
              <div className="leading-tight">
                <div className="text-[14px] font-semibold tracking-tight">{p.title}</div>
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
                  saros body · /{p.id}
                </div>
              </div>
            </div>

            <nav className="flex items-center gap-1 text-[13px]">
              {p.sections.map((s) => (
                <a key={s.href} href={s.href} className="rounded-md px-3 py-1.5 text-ink-dim hover:bg-white/5">
                  {s.label}
                </a>
              ))}
              <Link
                href="/"
                className="ml-2 rounded-md border border-line bg-void-3 px-3 py-1.5 font-medium text-ink transition hover:bg-white/5"
              >
                ← Hub
              </Link>
            </nav>
          </div>
        </header>

        <main className="relative z-10">
          <div className="mx-auto max-w-6xl px-5">
            <section className="pt-12 pb-8">
              <p
                className="font-mono text-[11px] uppercase tracking-[0.24em]"
                style={{ color: p.accent }}
              >
                saros body · {p.id}
              </p>
              <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
                {p.tagline}
              </h1>
            </section>
          </div>
          {children}
        </main>

        <WakeIndicator />

        <footer className="relative z-10 mt-16 border-t border-line px-5 py-8">
          <div className="mx-auto max-w-6xl text-[12px] text-ink-faint">
            An independent Saros body. Own route, own identity, own models, own tests — connected
            to the hub only through <code className="font-mono">core/contract.py</code>.
          </div>
        </footer>
      </body>
    </html>
  );
}
