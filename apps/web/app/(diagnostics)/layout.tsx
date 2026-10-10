import type { Metadata, Viewport } from "next";

import {
  Storefront,
  storefrontMetadata,
  storefrontViewport,
  type StorefrontProps,
} from "@/components/body/Storefront";

const P: StorefrontProps = {
  id: "xai_health",
  title: "Explainable Diagnostics",
  tagline: "Predict the risk, explain the risk, and audit who the model fails.",
  accent: "#a78bfa",
  sections: [{ href: "#model", label: "Model" }, { href: "#console", label: "Console" }],
};

export const metadata: Metadata = storefrontMetadata(P);

export const viewport: Viewport = storefrontViewport(P);

export default function Layout({ children }: { children: React.ReactNode }) {
  return <Storefront p={P}>{children}</Storefront>;
}
