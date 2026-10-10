import type { Metadata, Viewport } from "next";

import {
  Storefront,
  storefrontMetadata,
  storefrontViewport,
  type StorefrontProps,
} from "@/components/body/Storefront";

const P: StorefrontProps = {
  id: "rag",
  title: "Knowledge RAG",
  tagline: "Answers grounded in the corpus, with the source attached to every claim.",
  accent: "#38bdf8",
  sections: [{ href: "#model", label: "Model" }, { href: "#console", label: "Console" }],
};

export const metadata: Metadata = storefrontMetadata(P);

export const viewport: Viewport = storefrontViewport(P);

export default function Layout({ children }: { children: React.ReactNode }) {
  return <Storefront p={P}>{children}</Storefront>;
}
