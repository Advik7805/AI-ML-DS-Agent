import type { Metadata, Viewport } from "next";

import {
  Storefront,
  storefrontMetadata,
  storefrontViewport,
  type StorefrontProps,
} from "@/components/body/Storefront";

const P: StorefrontProps = {
  id: "scraper",
  title: "Web Scraper",
  tagline: "Turn any page into data the other bodies can consume.",
  accent: "#f97316",
  sections: [{ href: "#model", label: "Model" }, { href: "#console", label: "Console" }],
};

export const metadata: Metadata = storefrontMetadata(P);

export const viewport: Viewport = storefrontViewport(P);

export default function Layout({ children }: { children: React.ReactNode }) {
  return <Storefront p={P}>{children}</Storefront>;
}
