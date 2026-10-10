import type { Metadata, Viewport } from "next";

import {
  Storefront,
  storefrontMetadata,
  storefrontViewport,
  type StorefrontProps,
} from "@/components/body/Storefront";

const P: StorefrontProps = {
  id: "pricing",
  title: "Dynamic Pricing",
  tagline: "Price from measured elasticity — never from an extrapolated guess.",
  accent: "#34d399",
  sections: [
    { href: "#board", label: "Opportunities" },
    { href: "#console", label: "Console" },
  ],
};

export const metadata: Metadata = storefrontMetadata(P);

export const viewport: Viewport = storefrontViewport(P);

export default function Layout({ children }: { children: React.ReactNode }) {
  return <Storefront p={P}>{children}</Storefront>;
}
