import type { Metadata, Viewport } from "next";
import Link from "next/link";

import "@/app/globals.css";

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
    title: `${p.title} — Nexus`,
    description: `${p.tagline} An independent Nexus body connected to the hub through the shared module contract.`,
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
      <body className="min-h-screen bg-[#fbfaf7] text-[#1c1917] antialiased">
        <StorefrontField accent={p.accent} />

        <header className="sticky top-0 z-30 border-b border-[#e7e0d5] bg-[#fbfaf7]/85 backdrop-blur">
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
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#a8a29e]">
                  nexus body · /{p.id}
                </div>
              </div>
            </div>

            <nav className="flex items-center gap-1 text-[13px]">
              {p.sections.map((s) => (
                <a key={s.href} href={s.href} className="rounded-md px-3 py-1.5 text-[#57534e] hover:bg-black/5">
                  {s.label}
                </a>
              ))}
              <Link
                href="/"
                className="ml-2 rounded-md border border-[#e7e0d5] bg-white px-3 py-1.5 font-medium text-[#292524] transition hover:bg-[#fafaf9]"
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
                nexus body · {p.id}
              </p>
              <h1 className="mt-4 max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
                {p.tagline}
              </h1>
            </section>
          </div>
          {children}
        </main>

        <footer className="relative z-10 mt-16 border-t border-[#e7e0d5] px-5 py-8">
          <div className="mx-auto max-w-6xl text-[12px] text-[#a8a29e]">
            An independent Nexus body. Own route, own identity, own models, own tests — connected
            to the hub only through <code className="font-mono">core/contract.py</code>.
          </div>
        </footer>
      </body>
    </html>
  );
}
