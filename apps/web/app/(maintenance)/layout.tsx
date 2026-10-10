import type { Metadata, Viewport } from "next";

import {
  Storefront,
  storefrontMetadata,
  storefrontViewport,
  type StorefrontProps,
} from "@/components/body/Storefront";

const P: StorefrontProps = {
  id: "maintenance",
  title: "Predictive Maintenance",
  tagline: "Know which engine fails next, and how many cycles you have to act.",
  accent: "#22d3ee",
  sections: [{ href: "#model", label: "Model" }, { href: "#console", label: "Console" }],
};

export const metadata: Metadata = storefrontMetadata(P);

export const viewport: Viewport = storefrontViewport(P);

export default function Layout({ children }: { children: React.ReactNode }) {
  return <Storefront p={P}>{children}</Storefront>;
}
