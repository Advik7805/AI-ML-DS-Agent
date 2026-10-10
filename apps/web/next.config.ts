import type { NextConfig } from "next";

/**
 * The browser must never call the backend directly — it only ever speaks to this origin, and
 * Next proxies to FastAPI. That keeps the preview host working and avoids CORS in dev.
 *
 * `API_ORIGIN` points at the local FastAPI in dev and at the deployed backend in production.
 */
const API_ORIGIN = process.env.API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // Host-based routing: one deploy, many storefronts. Each body can later be given its
      // own subdomain (churn.nexus.dev) that renders a completely different root layout,
      // exactly like a retailer pointing /electronics at a separate storefront.
      ...(process.env.CHURN_HOST
        ? [
            {
              source: "/:path*",
              has: [{ type: "host" as const, value: process.env.CHURN_HOST }],
              destination: "/churn/:path*",
            },
          ]
        : []),
      { source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` },
      { source: "/health", destination: `${API_ORIGIN}/health` },
    ];
  },
};

export default nextConfig;
