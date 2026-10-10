import type { Metadata, Viewport } from "next";

import {
  Storefront,
  storefrontMetadata,
  storefrontViewport,
  type StorefrontProps,
} from "@/components/body/Storefront";

const P: StorefrontProps = {
  id: "fraud",
  title: "Fraud & Anomaly",
  tagline: "Score every transaction before it settles — and catch the fraud you have never seen.",
  accent: "#f43f5e",
  sections: [{ href: "#model", label: "Model" }, { href: "#console", label: "Console" }],
};

export const metadata: Metadata = storefrontMetadata(P);

export const viewport: Viewport = storefrontViewport(P);

export default function Layout({ children }: { children: React.ReactNode }) {
  return <Storefront p={P}>{children}</Storefront>;
}
